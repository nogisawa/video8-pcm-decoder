use std::{
    env,
    fs::File,
    io::{self, BufWriter, Write},
    path::PathBuf,
    process,
};
use v8pcm::{
    BLOCKS_PER_FIELD,
    demod::{NTSC_BIT_RATE, PAL_BIT_RATE, field_starts, hard_block_at, scan_window},
    sndfile::SoundFile,
};

fn help() {
    println!(
        r#"v8demod - demodulate a CXADC FLAC capture to Video8 PCM blocks

Usage: v8demod [OPTIONS] INPUT.flac

Options:
  -o, --output FILE       output file; default is stdout
  --sample-rate HZ        actual ADC rate; default is FLAC metadata (x1000 below 1 MHz)
  --system ntsc|pal       television system (default: ntsc)
  --phases N              clock phases per window (default: 12)
  --window-ms MS          analysis window (default: 8)
  --overlap-ms MS         overlap, must cover a PCM field (default: 4)
  --start SECONDS         actual RF start time (default: 0)
  --duration SECONDS      optional actual RF duration
  -h, --help              show this help

Output is headerless physical-order 13-byte blocks."#
    );
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() == 1 || args.iter().any(|x| x == "-h" || x == "--help") {
        help();
        return Ok(());
    }
    let (mut output, mut rate, mut system, mut phases): (
        Option<String>,
        Option<f64>,
        String,
        usize,
    ) = (None, None, "ntsc".to_string(), 12usize);
    let (mut window_ms, mut overlap_ms, mut start, mut duration): (f64, f64, f64, Option<f64>) =
        (8.0, 4.0, 0.0, None);
    let mut input = None;
    let mut i = 1;
    while i < args.len() {
        let value = |i: &mut usize| -> String {
            *i += 1;
            args.get(*i).cloned().unwrap_or_default()
        };
        match args[i].as_str() {
            "-o" | "--output" => output = Some(value(&mut i)),
            "--sample-rate" => rate = value(&mut i).parse().ok(),
            "--system" => system = value(&mut i),
            "--phases" => phases = value(&mut i).parse().unwrap_or(0),
            "--window-ms" => window_ms = value(&mut i).parse().unwrap_or(0.0),
            "--overlap-ms" => overlap_ms = value(&mut i).parse().unwrap_or(0.0),
            "--start" => start = value(&mut i).parse().unwrap_or(-1.0),
            "--duration" => duration = value(&mut i).parse().ok(),
            x if x.starts_with('-') => {
                eprintln!("unknown option: {x}");
                process::exit(2);
            }
            x => input = Some(PathBuf::from(x)),
        }
        i += 1;
    }
    if phases == 0
        || window_ms <= overlap_ms
        || overlap_ms < 2.6
        || start < 0.0
        || !matches!(system.as_str(), "ntsc" | "pal")
    {
        eprintln!("invalid parameters; use --help");
        process::exit(2);
    }
    let mut source = SoundFile::open(
        &input.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "missing INPUT"))?,
    )?;
    let metadata_rate = source.sample_rate as f64;
    let sample_rate = rate.unwrap_or(if metadata_rate < 1_000_000.0 {
        metadata_rate * 1000.0
    } else {
        metadata_rate
    });
    let bit_rate = if system == "ntsc" {
        NTSC_BIT_RATE
    } else {
        PAL_BIT_RATE
    };
    let blocks_per_field = if system == "ntsc" { 132 } else { 157 };
    eprintln!(
        "system={system} sample_rate={sample_rate:.3}Hz metadata_rate={metadata_rate:.3}Hz{}",
        if rate.is_none() { " (automatic)" } else { "" }
    );
    if blocks_per_field != BLOCKS_PER_FIELD {
        return Err(io::Error::other("PAL output is not implemented yet"));
    }
    let window = (window_ms * 0.001 * sample_rate).round() as usize;
    let step = ((window_ms - overlap_ms) * 0.001 * sample_rate).round() as i64;
    let first = (start * sample_rate).round() as i64;
    let last = duration
        .map(|d| first + (d * sample_rate).round() as i64)
        .unwrap_or(source.frames)
        .min(source.frames);
    let sink: Box<dyn Write> = match output {
        Some(p) => Box::new(File::create(p)?),
        None => Box::new(io::stdout()),
    };
    let mut sink = BufWriter::new(sink);
    let spb = sample_rate / bit_rate;
    let block_step = 107.0 * spb;
    let mut position = first;
    let mut last_field = -1e30_f64;
    let mut fields = 0usize;
    while position < last {
        let samples = source.read_mono(position, window.min((last - position) as usize))?;
        if samples.len() < 1000 {
            break;
        }
        let (filtered, candidates) = scan_window(&samples, position, sample_rate, bit_rate, phases);
        let unfiltered: Vec<f64> = samples.iter().map(|&value| value as f64).collect();
        for (field_zero, count) in field_starts(&candidates, sample_rate, bit_rate) {
            if field_zero - last_field < sample_rate * 0.010 {
                continue;
            }
            let relative = field_zero - position as f64;
            if relative < 8.0
                || relative + blocks_per_field as f64 * block_step + 104.0 * spb
                    >= filtered.len() as f64
            {
                continue;
            }
            let mut known: Vec<Option<[u8; 13]>> = vec![None; blocks_per_field];
            for candidate in &candidates {
                let estimate = candidate.sample - candidate.address as f64 * block_step;
                if (estimate - field_zero).abs() < 8.0 {
                    known[candidate.address as usize] = Some(candidate.raw);
                }
            }
            for address in 0..blocks_per_field {
                let raw = known[address].unwrap_or_else(|| {
                    let location = relative + address as f64 * block_step;
                    let first = hard_block_at(&filtered, location, sample_rate, bit_rate);
                    let recorded = u16::from_le_bytes([first[11], first[12]]);
                    if v8pcm::crc16_video8(&first[..11]) == recorded {
                        first
                    } else {
                        hard_block_at(&unfiltered, location, sample_rate, bit_rate)
                    }
                });
                sink.write_all(&raw)?;
            }
            last_field = field_zero;
            fields += 1;
            eprintln!("field={fields} sample={field_zero:.1} seed_crc_blocks={count}");
        }
        position += step;
    }
    sink.flush()?;
    eprintln!(
        "wrote {fields} fields, {} blocks",
        fields * blocks_per_field
    );
    Ok(())
}
