from pathlib import Path

from demo.s1_config import app


def main():
    app.batch_process(
        src_dir=Path(__file__).parent / 'batch',
        dst_dir=Path(__file__).parent / 'batch-output',
        pixel_count=1250,
    )


if __name__ == '__main__':
    main()
