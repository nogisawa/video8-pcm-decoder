use std::{
    collections::BTreeMap,
    env,
    fs::File,
    io::{self, BufWriter, Write},
    path::PathBuf,
    process,
    sync::{Arc, Mutex, mpsc},
    thread,
};
use v8pcm::{
    BLOCKS_PER_FIELD,
    demod::{
        Candidate, NTSC_BIT_RATE, PAL_BIT_RATE, field_starts, hard_block_at,
        scan_window_gated_with_workers,
    },
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
  --gate RATIO            skip low-activity windows (default: 0.12; 0 disables)
  --threads N             parallel analysis workers (default: logical CPUs)
  --buffer-windows N      decoded input windows kept in flight (default: 2 x threads)
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
    let mut gate = 0.12_f64;
    let mut threads = thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1);
    let mut buffer_windows = None;
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
            "--gate" => gate = value(&mut i).parse().unwrap_or(-1.0),
            "--threads" => threads = value(&mut i).parse().unwrap_or(0),
            "--buffer-windows" => buffer_windows = value(&mut i).parse().ok(),
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
        || gate < 0.0
        || threads == 0
        || start < 0.0
        || !matches!(system.as_str(), "ntsc" | "pal")
    {
        eprintln!("invalid parameters; use --help");
        process::exit(2);
    }
    let input =
        input.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "missing INPUT"))?;
    let source = SoundFile::open(&input)?;
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
    drop(source);
    let buffer_windows = buffer_windows.unwrap_or(threads.saturating_mul(2)).max(1);
    let sink: Box<dyn Write> = match output {
        Some(p) => Box::new(File::create(p)?),
        None => Box::new(io::stdout()),
    };
    let mut sink = BufWriter::new(sink);
    let spb = sample_rate / bit_rate;
    let block_step = 107.0 * spb;
    let mut last_field = -1e30_f64;
    let mut fields = 0usize;
    let jobs = if last - first < 1000 {
        0
    } else {
        ((last - 1000 - first) / step + 1) as usize
    };
    let (job_tx, job_rx) = mpsc::sync_channel::<(usize, i64, Vec<f32>)>(buffer_windows);
    let job_rx = Arc::new(Mutex::new(job_rx));
    let (result_tx, result_rx) = mpsc::sync_channel::<WindowResult>(buffer_windows);

    let reader_input = input.clone();
    let reader = thread::spawn(move || -> io::Result<()> {
        let mut source = SoundFile::open(&reader_input)?;
        for sequence in 0..jobs {
            let position = first + sequence as i64 * step;
            let samples = source.read_mono(position, window.min((last - position) as usize))?;
            job_tx
                .send((sequence, position, samples))
                .map_err(|_| io::Error::other("analysis workers stopped"))?;
        }
        Ok(())
    });

    let mut workers = Vec::with_capacity(threads);
    for _ in 0..threads {
        let job_rx = Arc::clone(&job_rx);
        let result_tx = result_tx.clone();
        workers.push(thread::spawn(move || {
            loop {
                let job = job_rx.lock().expect("job queue poisoned").recv();
                let Ok((sequence, position, samples)) = job else {
                    break;
                };
                let (filtered, candidates) = scan_window_gated_with_workers(
                    &samples,
                    position,
                    sample_rate,
                    bit_rate,
                    phases,
                    gate,
                    1,
                );
                if result_tx
                    .send(WindowResult {
                        sequence,
                        position,
                        samples,
                        filtered,
                        candidates,
                    })
                    .is_err()
                {
                    break;
                }
            }
        }));
    }
    drop(result_tx);

    let mut pending = BTreeMap::new();
    let mut next_sequence = 0usize;
    while next_sequence < jobs {
        let result = result_rx
            .recv()
            .map_err(|_| io::Error::other("analysis workers stopped early"))?;
        pending.insert(result.sequence, result);
        while let Some(result) = pending.remove(&next_sequence) {
            let WindowResult {
                position,
                samples,
                filtered,
                candidates,
                ..
            } = result;
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
            next_sequence += 1;
        }
    }
    reader
        .join()
        .map_err(|_| io::Error::other("FLAC reader panicked"))??;
    for worker in workers {
        worker
            .join()
            .map_err(|_| io::Error::other("analysis worker panicked"))?;
    }
    sink.flush()?;
    eprintln!(
        "wrote {fields} fields, {} blocks",
        fields * blocks_per_field
    );
    Ok(())
}

struct WindowResult {
    sequence: usize,
    position: i64,
    samples: Vec<f32>,
    filtered: Vec<f64>,
    candidates: Vec<Candidate>,
}
