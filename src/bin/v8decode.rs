use std::{
    env,
    fs::File,
    io::{self, BufReader, BufWriter, Read, Write},
    process,
};
use v8pcm::{
    FIELD_BYTES, ReconstructionFilter, WAV_RATE, conceal_field, decode_field, read_exact_or_eof,
    write_pcm,
};

fn main() -> io::Result<()> {
    let mut use_bad = false;
    let mut lowpass = 15_000.0_f64;
    let mut path = None;
    let args: Vec<_> = env::args().skip(1).collect();
    let mut at = 0;
    while at < args.len() {
        let arg = &args[at];
        match arg.as_str() {
            "-h" | "--help" => {
                println!(
                    r#"v8decode - convert Video8 PCM blocks to raw stereo PCM

Usage: v8decode [OPTIONS] [INPUT|-] > audio.s16le

Input is fixed 13-byte NTSC blocks; stdin is used when omitted or '-'.
Output is headerless signed 16-bit little-endian, 2 channels, 31469 Hz.

Options:
  --use-bad-crc  decode CRC-bad blocks instead of concealment
  --lowpass HZ    reconstruction low-pass (default: 15000; 0 disables)
  -h, --help       show this help

Example:
  v8decode pcm.bin | ffmpeg -f s16le -ar 31469 -ac 2 -i - output.wav"#
                );
                return Ok(());
            }
            "--use-bad-crc" => use_bad = true,
            "--lowpass" => {
                at += 1;
                lowpass = args.get(at).and_then(|x| x.parse().ok()).unwrap_or(-1.0);
            }
            "-" => path = None,
            _ if arg.starts_with('-') => {
                eprintln!("unknown option: {arg}");
                process::exit(2);
            }
            _ => path = Some(arg),
        }
        at += 1;
    }
    if !(0.0..WAV_RATE as f64 / 2.0).contains(&lowpass) && lowpass != 0.0 {
        eprintln!("--lowpass must be 0 or below {} Hz", WAV_RATE / 2);
        process::exit(2);
    }
    let input: Box<dyn Read> = match path {
        Some(p) => Box::new(File::open(p)?),
        None => Box::new(io::stdin()),
    };
    let mut input = BufReader::new(input);
    let mut output = BufWriter::new(io::stdout());
    let mut field = [0_u8; FIELD_BYTES];
    let mut reconstruction = (lowpass > 0.0).then(|| ReconstructionFilter::new(lowpass));
    let (mut fields, mut concealed) = (0_usize, 0_usize);
    while read_exact_or_eof(&mut input, &mut field)? {
        let mut audio = decode_field(&field, use_bad);
        concealed += conceal_field(&mut audio);
        if let Some(filter) = &mut reconstruction {
            filter.process_field(&mut audio);
        }
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
