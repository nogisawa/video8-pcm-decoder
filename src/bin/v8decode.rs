use std::{
    env,
    fs::File,
    io::{self, BufReader, BufWriter, Read, Write},
    process,
};
use v8pcm::{FIELD_BYTES, conceal_field, decode_field, read_exact_or_eof, write_pcm};

fn main() -> io::Result<()> {
    let mut use_bad = false;
    let mut path = None;
    for arg in env::args().skip(1) {
        match arg.as_str() {
            "-h" | "--help" => {
                println!(
                    r#"v8decode - convert Video8 PCM blocks to raw stereo PCM

Usage: v8decode [OPTIONS] [INPUT|-] > audio.s16le

Input is fixed 13-byte NTSC blocks; stdin is used when omitted or '-'.
Output is headerless signed 16-bit little-endian, 2 channels, 31469 Hz.

Options:
  --use-bad-crc  decode CRC-bad blocks instead of concealment
  -h, --help       show this help

Example:
  v8decode pcm.bin | ffmpeg -f s16le -ar 31469 -ac 2 -i - output.wav"#
                );
                return Ok(());
            }
            "--use-bad-crc" => use_bad = true,
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
    let mut field = [0_u8; FIELD_BYTES];
    let (mut fields, mut concealed) = (0_usize, 0_usize);
    while read_exact_or_eof(&mut input, &mut field)? {
        let mut audio = decode_field(&field, use_bad);
        concealed += conceal_field(&mut audio);
        write_pcm(&audio, &mut output)?;
        fields += 1;
    }
    output.flush()?;
    eprintln!(
        "fields={fields} frames={} concealed_channel_samples={concealed}",
        fields * 525
    );
    Ok(())
}
