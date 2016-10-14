use clap::{App, ArgMatches, Arg, SubCommand, AppSettings};
use std::{f32, usize};
use std::str::FromStr;

fn validator<T: FromStr>(tok: String) -> Result<(), String> {
  match T::from_str(&tok) {
    Ok(_) => Ok(()),
    Err(_) => Err(format!("malformed value: {}", tok)),
  }
}

fn dist_def_validator(val: String) -> Result<(), String> {
  let mut tokens = val.split(',');
  let dname;
  match tokens.next() {
    Some("normal") => dname = "normal",
    Some("uniform") => dname = "uniform",
    None | Some(_) => return Err("distribution name malformed".to_string()),
  }
  let min_or_mean;
  match f32::from_str(tokens.next().unwrap()) {
    Ok(x) => min_or_mean = x,
    Err(_) => return Err("distribution parameter 1 malformed".to_string()),
  }
  match f32::from_str(tokens.next().unwrap()) {
    Ok(max_or_stddev) => {
      if dname == "uniform" && min_or_mean > max_or_stddev {
        return Err("malformed uniform distribution specified".to_string());
      } else if dname == "normal" && max_or_stddev < 0.0 {
        return Err("malformed normal distribution specified".to_string());
      }
    },
    Err(_) => return Err("distribution parameter 2 malformed".to_string()),
  }

  Ok(())
}

pub fn get<'a>() -> ArgMatches<'a> {
  let train_args = vec![
    Arg::with_name("INPUT")
      .index(1)
      .required(true)
      .help("Path to CSV file with training set"),
    Arg::with_name("output")
      .short("o")
      .long("output")
      .takes_value(true)
      .help("Path to output file with trained model (stdout if not given)"),
    Arg::with_name("init_dist")
      .long("init-dist")
      .takes_value(true)
      .default_value("normal,0.0,1.0")
      .validator(dist_def_validator)
      .help("Specifies the distribution from which initial weights will be sampled (format: 'name,param1,param2', allowed names: 'normal', 'uniform')"),
    Arg::with_name("train_rate")
      .long("alpha")
      .short("a")
      .takes_value(true)
      .validator(validator::<f32>)
      .default_value("0.1")
      .help("Training rate"),
    Arg::with_name("bipolar")
      .long("bipolar")
      .help("Use bipolar logic (default is unipolar)"),
    Arg::with_name("epochs")
      .long("epochs")
      .short("e")
      .takes_value(true)
      .validator(validator::<usize>)
      .default_value("5")
      .help("Maximum number of epochs (training may be terminated earlier if the model can't be improved further)")
  ];

  App::new("perc")
    .author("Maciej Śniegocki")
    .setting(AppSettings::SubcommandRequired)
    .subcommand(SubCommand::with_name("train")
      .args(&train_args))
    .subcommand(SubCommand::with_name("train-ada")
      .args(&train_args)
      .arg(Arg::with_name("termination_threshold")
        .default_value("1.0")
        .takes_value(true)
        .validator(validator::<f32>)
        .long("threshold")
        .short("t")
        .help("Error at which training is allowed to terminate")))
    .subcommand(SubCommand::with_name("eval")
      .arg(Arg::with_name("MODEL")
        .index(1)
        .required(true)
        .help("Path to model being evaluated"))
      .arg(Arg::with_name("X1")
        .index(2)
        .required(true)
        .validator(validator::<f32>)
        .help("First input"))
      .arg(Arg::with_name("X2")
        .index(3)
        .required(true)
        .validator(validator::<f32>)
        .help("Second input")))
      .arg(Arg::with_name("bipolar")
        .long("bipolar")
        .help("Use bipolar logic (default is unipolar)"))
    .subcommand(SubCommand::with_name("gen")
      .arg(Arg::with_name("func")
        .index(1)
        .required(true)
        .possible_values(&["or", "xor", "and"])
        .help("Logical function to be generated"))
      .arg(Arg::with_name("samples")
        .index(2)
        .validator(validator::<usize>)
        .default_value("100")
        .help("Number of samples to be generated"))
      .arg(Arg::with_name("noise_amt")
        .long("noise-amt")
        .validator(validator::<f32>)
        .default_value("0.0")
        .takes_value(true)
        .help("Amount of noise as percentage of all samples"))
      .arg(Arg::with_name("sigma")
        .long("sigma")
        .short("s")
        .validator(validator::<f32>)
        .default_value("0.0")
        .takes_value(true)
        .help("Generate noisy samples with the given standard deviation"))
      .arg(Arg::with_name("bipolar")
        .long("bipolar")
        .help("Use bipolar logic (default is unipolar)")))
    .subcommand(SubCommand::with_name("validate")
      .arg(Arg::with_name("bipolar")
        .long("bipolar")
        .help("Use bipolar logic (default is unipolar)"))
      .arg(Arg::with_name("model")
        .index(1)
        .required(true)
        .help("Model to be validated"))
      .arg(Arg::with_name("input")
        .index(2)
        .required(true)
        .help("Validation data")))
    .get_matches()
}
