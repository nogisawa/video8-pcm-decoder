use std::{collections::HashSet, env, io, path::PathBuf, process};
use v8pcm::{
    demod::{NTSC_BIT_RATE, PAL_BIT_RATE, field_starts, scan_window},
    sndfile::SoundFile,
};

#[derive(Clone)]
struct ResultRow {
    system: &'static str,
    rate: f64,
    blocks: usize,
    fields: usize,
}
impl ResultRow {
    fn score(&self) -> usize {
        self.blocks * 100 + self.fields
    }
}

fn help() {
    println!(
        r#"v8tune - find Video8 demodulation parameters using CRC results

Usage: v8tune [OPTIONS] INPUT.flac

Options:
  --system auto|ntsc|pal  standards to test (default: auto)
  --sample-rate HZ        search centre; metadata or metadata x1000 by default
  --range-percent PCT     initial range on each side (default: 1)
  --steps N               candidates per pass (default: 9)
  --refine-passes N       successive searches around winner (default: 2)
  --duration SECONDS      RF data tested (default: 0.05)
  --start SECONDS         RF start position (default: 0)
  --phases N              bit phases per candidate (default: 8)
  --window-ms MS          analysis window (default: 8)
  -h, --help              show this help"#
    );
}

fn evaluate(
    samples: &[f32],
    rate: f64,
    system: &'static str,
    phases: usize,
    window_ms: f64,
) -> ResultRow {
    let bit_rate = if system == "ntsc" {
        NTSC_BIT_RATE
    } else {
        PAL_BIT_RATE
    };
    let window = (window_ms * 0.001 * rate).round() as usize;
    let step = (window as f64 - rate * 0.0006).round() as usize;
    let spb = rate / bit_rate;
    let mut unique = HashSet::new();
    let mut starts = Vec::new();
    let mut position = 0usize;
    while position < samples.len() {
        let end = (position + window).min(samples.len());
        if end - position < 1000 {
            break;
        }
        let (_, candidates) = scan_window(
            &samples[position..end],
            position as i64,
            rate,
            bit_rate,
            phases,
        );
        for candidate in &candidates {
            unique.insert(((candidate.sample / spb).round() as i64, candidate.address));
        }
        starts.extend(
            field_starts(&candidates, rate, bit_rate)
                .into_iter()
                .map(|x| x.0),
        );
        position += step.max(1);
    }
    starts.sort_by(|a, b| a.total_cmp(b));
    starts.dedup_by(|a, b| (*a - *b).abs() < rate * 0.010);
    ResultRow {
        system,
        rate,
        blocks: unique.len(),
        fields: starts.len(),
    }
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() == 1 || args.iter().any(|x| x == "-h" || x == "--help") {
        help();
        return Ok(());
    }
    let (mut system, mut rate) = ("auto".to_string(), None::<f64>);
    let (mut range, mut steps, mut passes) = (1.0, 9usize, 2usize);
    let (mut duration, mut start, mut phases, mut window_ms) = (0.05, 0.0, 8usize, 8.0);
    let mut input = None;
    let mut i = 1;
    while i < args.len() {
        let value = |i: &mut usize| {
            *i += 1;
            args.get(*i).cloned().unwrap_or_default()
        };
        match args[i].as_str() {
            "--system" => system = value(&mut i),
            "--sample-rate" => rate = value(&mut i).parse().ok(),
            "--range-percent" => range = value(&mut i).parse().unwrap_or(0.0),
            "--steps" => steps = value(&mut i).parse().unwrap_or(0),
            "--refine-passes" => passes = value(&mut i).parse().unwrap_or(0),
            "--duration" => duration = value(&mut i).parse().unwrap_or(0.0),
            "--start" => start = value(&mut i).parse().unwrap_or(-1.0),
            "--phases" => phases = value(&mut i).parse().unwrap_or(0),
            "--window-ms" => window_ms = value(&mut i).parse().unwrap_or(0.0),
            x if x.starts_with('-') => {
                eprintln!("unknown option: {x}");
                process::exit(2)
            }
            x => input = Some(PathBuf::from(x)),
        }
        i += 1;
    }
    if steps < 3
        || passes == 0
        || phases == 0
        || range <= 0.0
        || duration <= 0.0
        || start < 0.0
        || !matches!(system.as_str(), "auto" | "ntsc" | "pal")
    {
        eprintln!("invalid parameters; use --help");
        process::exit(2)
    }
    let mut source = SoundFile::open(
        &input.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "missing INPUT"))?,
    )?;
    let metadata = source.sample_rate as f64;
    let centre = rate.unwrap_or(if metadata < 1e6 {
        metadata * 1000.0
    } else {
        metadata
    });
    let first = (start * centre).round() as i64;
    let count = (duration * centre * (1.0 + range / 100.0)).round() as usize;
    let samples = source.read_mono(first, count)?;
    let mut search_centre = centre;
    let mut half = centre * range / 100.0;
    let mut systems: Vec<&'static str> = match system.as_str() {
        "auto" => vec!["ntsc", "pal"],
        "ntsc" => vec!["ntsc"],
        _ => vec!["pal"],
    };
    let mut all = Vec::new();
    for pass in 0..passes {
        println!(
            "\nPass {}/{}: {:.3} +/- {:.3} Hz",
            pass + 1,
            passes,
            search_centre,
            half
        );
        let mut rows = Vec::new();
        for &standard in &systems {
            for n in 0..steps {
                let candidate = search_centre - half + 2.0 * half * n as f64 / (steps - 1) as f64;
                let row = evaluate(&samples, candidate, standard, phases, window_ms);
                println!(
                    "{standard:4} {candidate:14.3} Hz  CRC blocks={:5} fields={}",
                    row.blocks, row.fields
                );
                rows.push(row);
            }
        }
        let winner = rows.iter().max_by_key(|r| r.score()).unwrap().clone();
        all.extend(rows);
        search_centre = winner.rate;
        half = 2.0 * half / (steps - 1) as f64;
        systems = vec![winner.system];
    }
    all.sort_by_key(|r| std::cmp::Reverse(r.score()));
    let best = &all[0];
    println!(
        "\nRecommended:\nv8demod INPUT.flac --system {} --sample-rate {:.3} --phases {}",
        best.system,
        best.rate,
        phases.max(12)
    );
    if best.blocks == 0 {
        process::exit(2)
    }
    Ok(())
}
