use std::{
    env,
    fs::File,
    io::{self, BufReader, BufWriter, Seek, SeekFrom, Write},
    process,
};
use v8pcm::{FIELD_BYTES, WAV_RATE, conceal_field, decode_field, read_exact_or_eof, write_pcm};

fn header(data_bytes: u32) -> [u8; 44] {
    let mut h = [0u8; 44];
    h[0..4].copy_from_slice(b"RIFF");
    h[4..8].copy_from_slice(&(36 + data_bytes).to_le_bytes());
    h[8..12].copy_from_slice(b"WAVE");
    h[12..16].copy_from_slice(b"fmt ");
    h[16..20].copy_from_slice(&16u32.to_le_bytes());
    h[20..22].copy_from_slice(&1u16.to_le_bytes());
    h[22..24].copy_from_slice(&2u16.to_le_bytes());
    h[24..28].copy_from_slice(&WAV_RATE.to_le_bytes());
    h[28..32].copy_from_slice(&(WAV_RATE * 4).to_le_bytes());
    h[32..34].copy_from_slice(&4u16.to_le_bytes());
    h[34..36].copy_from_slice(&16u16.to_le_bytes());
    h[36..40].copy_from_slice(b"data");
    h[40..44].copy_from_slice(&data_bytes.to_le_bytes());
    h
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() == 1 || args.iter().any(|x| x == "-h" || x == "--help") {
        println!(
            "v8wav - convert Video8 blocks directly to WAV\n\nUsage: v8wav [--use-bad-crc] INPUT.bin OUTPUT.wav"
        );
        return Ok(());
    }
    let use_bad = args.iter().any(|x| x == "--use-bad-crc");
    let positional: Vec<_> = args
        .iter()
        .skip(1)
        .filter(|x| !x.starts_with('-'))
        .collect();
    if positional.len() != 2 {
        eprintln!("expected INPUT and OUTPUT; use --help");
        process::exit(2)
    }
    let mut input = BufReader::new(File::open(positional[0])?);
    let file = File::create(positional[1])?;
    let mut output = BufWriter::new(file);
    output.write_all(&[0u8; 44])?;
    let mut field = [0u8; FIELD_BYTES];
    let (mut fields, mut concealed) = (0usize, 0usize);
    while read_exact_or_eof(&mut input, &mut field)? {
        let mut audio = decode_field(&field, use_bad);
        concealed += conceal_field(&mut audio);
        write_pcm(&audio, &mut output)?;
        fields += 1;
    }
    output.flush()?;
    let data_bytes = (fields * 525 * 4) as u32;
    output.get_mut().seek(SeekFrom::Start(0))?;
    output.get_mut().write_all(&header(data_bytes))?;
    eprintln!(
        "fields={fields} frames={} concealed_channel_samples={concealed}",
        fields * 525
    );
    Ok(())
}
