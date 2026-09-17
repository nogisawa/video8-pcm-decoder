use std::{
    env,
    fs::File,
    io::{self, BufReader, BufWriter, Read, Write},
    process,
};
use v8pcm::{BLOCK_BYTES, BLOCKS_PER_FIELD, crc16_video8};

fn main() -> io::Result<()> {
    let mut passthrough = false;
    let mut show_bad = false;
    let mut path = None;
    for arg in env::args().skip(1) {
        match arg.as_str() {
            "-h" | "--help" => {
                println!(
                    r#"v8crc - check headerless Video8 PCM blocks

Usage: v8crc [OPTIONS] [INPUT|-]

INPUT consists of fixed 13-byte physical-order blocks. Stdin is used when
INPUT is omitted or '-'. Statistics are written to stderr.

Options:
  --passthrough  copy every input block unchanged to stdout
  --show-bad     print every CRC failure to stderr
  -h, --help     show this help

Exit status is 1 when any CRC is bad, after all data is processed."#
                );
                return Ok(());
            }
            "--passthrough" => passthrough = true,
            "--show-bad" => show_bad = true,
            "-" => path = None,
            _ if arg.starts_with('-') => {
                eprintln!("unknown option: {arg}");
                process::exit(2);
            }
            _ => path = Some(arg),
        }
    }
    let input: Box<dyn Read> = match path {
        Some(p) => Box::new(File::open(p)?),
        None => Box::new(io::stdin()),
    };
    let mut input = BufReader::new(input);
    let mut output = BufWriter::new(io::stdout());
    let mut block = [0_u8; BLOCK_BYTES];
    let (mut total, mut good, mut address_errors) = (0_usize, 0_usize, 0_usize);
    loop {
        let mut filled = 0;
        while filled < BLOCK_BYTES {
            let n = input.read(&mut block[filled..])?;
            if n == 0 {
                break;
            }
            filled += n;
        }
        if filled == 0 {
            break;
        }
        if filled != BLOCK_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "partial block",
            ));
        }
        let expected = (total % BLOCKS_PER_FIELD) as u8;
        let recorded = u16::from_le_bytes([block[11], block[12]]);
        let calculated = crc16_video8(&block[..11]);
        let ok = recorded == calculated;
        good += ok as usize;
        address_errors += (block[0] != expected) as usize;
        if show_bad && !ok {
            eprintln!(
                "field={} block={} address={} recorded={:04x} calculated={:04x}",
                total / BLOCKS_PER_FIELD,
                expected,
                block[0],
                recorded,
                calculated
            );
        }
        if passthrough {
            output.write_all(&block)?;
        }
        total += 1;
    }
    output.flush()?;
    eprintln!(
        "blocks={total} crc_ok={good} crc_bad={} success={:.3}% address_errors={address_errors}",
        total - good,
        100.0 * good as f64 / total.max(1) as f64
    );
    if good != total {
        process::exit(1);
    }
    Ok(())
}
