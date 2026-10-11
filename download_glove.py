import argparse
import struct
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


DEFAULT_URL = "https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip"
TARGET_MEMBER = "glove.6B.100d.txt"
EOCD_SIGNATURE = b"PK\x05\x06"
CENTRAL_SIGNATURE = b"PK\x01\x02"
LOCAL_SIGNATURE = b"PK\x03\x04"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download only glove.6B.100d.txt from Stanford's ZIP file."
    )
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("embeddings/glove.6B.100d.txt"),
    )
    parser.add_argument("--parts", type=int, default=16)
    return parser.parse_args()


def get_remote_size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        return int(response.headers["Content-Length"])


def fetch_range(url: str, start: int, end: int, retries: int = 5) -> bytes:
    expected_size = end - start + 1
    last_error = None
    for _ in range(retries):
        try:
            request = urllib.request.Request(
                url,
                headers={"Range": f"bytes={start}-{end}"},
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read()
            if len(data) != expected_size:
                raise OSError(
                    f"Range {start}-{end} returned {len(data)} bytes, "
                    f"expected {expected_size}"
                )
            return data
        except (OSError, TimeoutError) as error:
            last_error = error
    raise OSError(f"Failed to download range {start}-{end}") from last_error


def find_member(url: str, remote_size: int) -> dict[str, int]:
    tail_size = min(remote_size, 65_557)
    tail_start = remote_size - tail_size
    tail = fetch_range(url, tail_start, remote_size - 1)
    eocd_position = tail.rfind(EOCD_SIGNATURE)
    if eocd_position < 0:
        raise ValueError("ZIP end-of-central-directory record not found")

    eocd = struct.unpack_from("<4s4H2LH", tail, eocd_position)
    central_size = eocd[5]
    central_offset = eocd[6]
    central = fetch_range(
        url,
        central_offset,
        central_offset + central_size - 1,
    )

    position = 0
    while position < len(central):
        header = struct.unpack_from("<4s6H3L5H2L", central, position)
        if header[0] != CENTRAL_SIGNATURE:
            raise ValueError("Invalid ZIP central-directory entry")
        compression = header[4]
        crc32 = header[7]
        compressed_size = header[8]
        uncompressed_size = header[9]
        filename_length = header[10]
        extra_length = header[11]
        comment_length = header[12]
        local_offset = header[16]
        name_start = position + 46
        name_end = name_start + filename_length
        filename = central[name_start:name_end].decode("utf-8")
        if filename == TARGET_MEMBER:
            local_header = fetch_range(url, local_offset, local_offset + 29)
            local = struct.unpack("<4s5H3L2H", local_header)
            if local[0] != LOCAL_SIGNATURE:
                raise ValueError("Invalid ZIP local-file header")
            data_offset = local_offset + 30 + local[9] + local[10]
            return {
                "compression": compression,
                "crc32": crc32,
                "compressed_size": compressed_size,
                "uncompressed_size": uncompressed_size,
                "data_offset": data_offset,
            }
        position = name_end + extra_length + comment_length

    raise FileNotFoundError(f"{TARGET_MEMBER} is not present in the remote ZIP")


def download_part(
    url: str,
    part_path: Path,
    start: int,
    end: int,
) -> Path:
    expected_size = end - start + 1
    if part_path.exists() and part_path.stat().st_size == expected_size:
        return part_path
    data = fetch_range(url, start, end)
    part_path.write_bytes(data)
    return part_path


def download_member(
    url: str,
    member: dict[str, int],
    output_path: Path,
    part_count: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    part_directory = output_path.parent / ".glove_parts"
    part_directory.mkdir(parents=True, exist_ok=True)

    compressed_size = member["compressed_size"]
    chunk_size = (compressed_size + part_count - 1) // part_count
    jobs = []
    with ThreadPoolExecutor(max_workers=part_count) as executor:
        for part_index in range(part_count):
            relative_start = part_index * chunk_size
            if relative_start >= compressed_size:
                break
            relative_end = min(relative_start + chunk_size, compressed_size) - 1
            part_path = part_directory / f"part-{part_index:03d}"
            jobs.append(
                executor.submit(
                    download_part,
                    url,
                    part_path,
                    member["data_offset"] + relative_start,
                    member["data_offset"] + relative_end,
                )
            )
        for completed_count, future in enumerate(as_completed(jobs), start=1):
            future.result()
            print(f"Downloaded {completed_count}/{len(jobs)} parts", flush=True)

    decompressor = (
        zlib.decompressobj(-zlib.MAX_WBITS)
        if member["compression"] == 8
        else None
    )
    if member["compression"] not in (0, 8):
        raise ValueError(f"Unsupported ZIP compression method: {member['compression']}")

    written_size = 0
    checksum = 0
    with output_path.open("wb") as output_file:
        for part_path in sorted(part_directory.glob("part-*")):
            with part_path.open("rb") as part_file:
                while chunk := part_file.read(1024 * 1024):
                    decoded = decompressor.decompress(chunk) if decompressor else chunk
                    output_file.write(decoded)
                    written_size += len(decoded)
                    checksum = zlib.crc32(decoded, checksum)
        if decompressor:
            decoded = decompressor.flush()
            output_file.write(decoded)
            written_size += len(decoded)
            checksum = zlib.crc32(decoded, checksum)

    if written_size != member["uncompressed_size"]:
        raise ValueError(
            f"Extracted size mismatch: {written_size} != {member['uncompressed_size']}"
        )
    if checksum & 0xFFFFFFFF != member["crc32"]:
        raise ValueError("Extracted GloVe file failed its CRC32 check")

    for part_path in part_directory.glob("part-*"):
        part_path.unlink()
    part_directory.rmdir()


def main() -> None:
    args = parse_args()
    if args.parts < 1:
        raise ValueError("--parts must be at least 1")
    if args.output.exists():
        print(f"GloVe file already exists: {args.output}")
        return

    remote_size = get_remote_size(args.url)
    member = find_member(args.url, remote_size)
    print(
        f"Downloading {TARGET_MEMBER}: "
        f"{member['compressed_size'] / 1024 / 1024:.1f} MiB compressed"
    )
    download_member(args.url, member, args.output, args.parts)
    print(f"Saved and verified {args.output}")


if __name__ == "__main__":
    main()
