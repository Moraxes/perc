extern crate clap;
extern crate rand;
extern crate time;

mod program_args;

use std::{f32, usize};
use std::str::FromStr;
use std::error::Error;
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

fn main() {
  let args = program_args::get();

  match args.subcommand_name() {
    Some("eval") => eval_cmd(args.subcommand_matches("eval").unwrap()),
    Some("train") => train_cmd(args.subcommand_matches("train").unwrap(), false),
    Some("train-ada") => train_cmd(args.subcommand_matches("train-ada").unwrap(), true),
    Some("gen") => gen_cmd(args.subcommand_matches("gen").unwrap()),
    Some("validate") => validate_cmd(args.subcommand_matches("validate").unwrap()),
    Some(_) | None => unreachable!(),
  }
}

type Model = (f32, f32, f32);

fn train_cmd<'a>(args: &clap::ArgMatches<'a>, adaline: bool) {
  let training_set_path = args.value_of("INPUT").unwrap();

  let s = file_to_string(training_set_path);
  let training_set = s.lines().map(|l| l.split_whitespace().map(|tok| f32::from_str(tok).unwrap()).collect::<Vec<f32>>()).collect::<Vec<_>>();
  let mut inputs = training_set.iter().map(|triple| (triple[0], triple[1])).collect::<Vec<(f32, f32)>>();
  let outputs = training_set.iter().map(|triple| triple[2]).collect::<Vec<f32>>();

  let model = train(args, &mut inputs[..], &outputs[..], adaline);

  match args.value_of("output") {
    None => println!("{} {} {}", model.0, model.1, model.2),
    Some(fname) => {
      let path = Path::new(fname);
      let display = path.display();
      let mut file = match File::create(&path) {
        Err(e) => panic!("couldn't open file {} for writing: {}", display, e.description()),
        Ok(f) => f,
      };
      match writeln!(file, "{} {} {}", model.0, model.1, model.2) {
        Err(e) => panic!("couldn't write to file {}: {}", display, e.description()),
        Ok(_) => {},
      }
    },
  }
}

fn get_rng() -> rand::XorShiftRng {
  use rand::*;
  use std::mem::transmute;
  XorShiftRng::from_seed(unsafe { transmute::<[u64; 2], [u32; 4]>([time::precise_time_ns(), time::precise_time_ns()]) })
}

fn train<'a>(args: &clap::ArgMatches<'a>, inputs: &mut [(f32, f32)], outputs: &[f32], adaline: bool) -> Model {
  let mut rng = get_rng();
  let mut weights = init_weights(args, &mut rng);
  let max_epochs = usize::from_str(args.value_of("epochs").unwrap()).unwrap();
  let train_rate = f32::from_str(args.value_of("train_rate").unwrap()).unwrap();
  let signed = args.is_present("bipolar");
  let adaline_threshold = f32::from_str(args.value_of("termination_threshold").unwrap_or("0.0")).unwrap();

  for epoch in 0..max_epochs {
    for (example, label) in inputs.iter().zip(outputs) {
      let out = if adaline {
        get_net(&weights[..], &[example.0, example.1])
      } else {
        eval(&weights[..], &[example.0, example.1], signed)
      };
      let err = (label - out).powi(2);
      println!("{}, {}, {}", out, label, err);
      weights[0] += train_rate * err * example.0;
      weights[1] += train_rate * err * example.1;
      weights[2] += train_rate * err;  // there is no example.2, it's the bias input
    }
    let mut total_err = 0f32;
    for (example, label) in inputs.iter().zip(outputs) {
      let out = if adaline {
        get_net(&weights[..], &[example.0, example.1])
      } else {
        eval(&weights[..], &[example.0, example.1], signed)
      };
      total_err += (label - out).powi(2);
    }
    println!("Epoch {}: {:?}, total error {}.", epoch, weights, total_err);

    if (adaline && total_err.abs() < adaline_threshold) || total_err == 0f32 {
      println!("Model cannot be improved; terminating.");
      break;
    }
  }

  (weights[0], weights[1], weights[2])
}

fn init_weights<'a, R: rand::Rng>(args: &clap::ArgMatches<'a>, rng: &mut R) -> Vec<f32> {
  use rand::distributions::*;

  let dist_def = args.value_of("init_dist").unwrap();
  let mut tokens = dist_def.split(',');
  let dname;
  match tokens.next() {
    Some("normal") => dname = "normal",
    Some("uniform") => dname = "uniform",
    None | Some(_) => dname = "",  // this will never happen but rustc complains
  }
  let min_or_mean = f32::from_str(tokens.next().unwrap()).unwrap();
  let max_or_stddev = f32::from_str(tokens.next().unwrap()).unwrap();
  let mut weights = vec![0.0, 0.0, 0.0];
  match dname {
    "normal" => {
      let dist = normal::Normal::new(min_or_mean as f64, max_or_stddev as f64);
      weights[0] = dist.ind_sample(rng) as f32;
      weights[1] = dist.ind_sample(rng) as f32;
      weights[2] = dist.ind_sample(rng) as f32;
    },
    "uniform" => {
      let dist = range::Range::new(min_or_mean as f64, max_or_stddev as f64);
      weights[0] = dist.ind_sample(rng) as f32;
      weights[1] = dist.ind_sample(rng) as f32;
      weights[2] = dist.ind_sample(rng) as f32;
    },
    _ => unreachable!(),
  }

  weights
}

fn eval_cmd<'a>(args: &clap::ArgMatches<'a>) {
  let model_path = args.value_of("MODEL").unwrap();
  let x1 = f32::from_str(args.value_of("X1").unwrap()).unwrap();
  let x2 = f32::from_str(args.value_of("X2").unwrap()).unwrap();
  let signed = args.is_present("bipolar");

  let s = file_to_string(model_path);

  let model: Vec<f32> = s.split_whitespace().map(|tok| f32::from_str(tok).unwrap()).collect();
  assert!(model.len() == 3);

  let inp = vec![x1, x2];
  println!("{}", eval(&model, &inp, signed));
}

fn get_net(model: &[f32], inp: &[f32]) -> f32 {
  model.last().unwrap() + inp.iter().zip(model.iter()).map(|(x1, x2)| x1 * x2).fold(0.0, |acc, el| acc + el)
}

fn eval(model: &[f32], inp: &[f32], signed: bool) -> f32 {
  if get_net(model, inp) > 0.0 { 1.0 } else if signed { -1.0 } else { 0.0 }
}

fn file_to_string(in_path: &str) -> String {
  let path = Path::new(in_path);
  let display = path.display();

  let mut file = match File::open(&path) {
      Err(e) => panic!("couldn't open {}: {}", display, e.description()),
      Ok(f) => f,
  };

  let mut s = String::new();
  match file.read_to_string(&mut s) {
      Err(e) => panic!("couldn't read {}: {}", display, e.description()),
      Ok(_) => {},
  }

  s
}

fn gen_cmd<'a>(args: &clap::ArgMatches<'a>) {
  use rand::*;
  use rand::distributions::IndependentSample;

  let mut rng = get_rng();
  let samples = usize::from_str(args.value_of("samples").unwrap()).unwrap();
  let func: Box<Fn(bool, bool) -> bool> = match args.value_of("func").unwrap() {
    "or" => Box::new(|a, b| a || b),
    "and" => Box::new(|a, b| a && b),
    "xor" => Box::new(|a, b| a ^ b),
    _ => unreachable!(),
  };

  if let Some(sigma_str) = args.value_of("sigma") {
    let sigma = f32::from_str(sigma_str).unwrap();
    let noisydist = distributions::Normal::new(0.0, sigma as f64);
    let noise_amt = f32::from_str(args.value_of("noise_amt").unwrap()).unwrap();
    let false_value = if args.is_present("bipolar") { -1.0 } else { 0.0 };
    for _ in 0..samples {
      let x1 = bool::rand(&mut rng);
      let x2 = bool::rand(&mut rng);
      let y = func(x1, x2);

      let mut x1_real = if x1 { 1.0 } else { false_value };
      let mut x2_real = if x2 { 1.0 } else { false_value };
      let y_real = if y { 1.0 } else { false_value };
      if rng.next_f32() < noise_amt {
        x1_real += noisydist.ind_sample(&mut rng) as f32;
        x2_real += noisydist.ind_sample(&mut rng) as f32;
      }

      println!("{} {} {}", x1_real, x2_real, y_real);
    }
  }
}

fn validate_cmd<'a>(args: &clap::ArgMatches<'a>) {
  let training_set_path = args.value_of("input").unwrap();

  let training_set_str = file_to_string(training_set_path);
  let mut training_set = training_set_str.lines().map(|l| l.split_whitespace().map(|tok| f32::from_str(tok).unwrap()).collect::<Vec<f32>>());
  let inputs = training_set.by_ref().map(|triple| (triple[0], triple[1])).collect::<Vec<(f32, f32)>>();
  let outputs = training_set.map(|triple| triple[2]).collect::<Vec<f32>>();

  let model_path = args.value_of("model").unwrap();
  let model_str = file_to_string(model_path);
  let model: Vec<f32> = model_str.split_whitespace().map(|tok| f32::from_str(tok).unwrap()).collect();

  let signed = args.is_present("bipolar");

  let mut errors = 0usize;
  for (&(x1, x2), &y) in inputs.iter().zip(outputs.iter()) {
    let model_result = eval(&model, &[x1, x2], signed);
    if model_result != y {
      errors += 1;
    }
  }
  println!("{} errors.", errors);
}