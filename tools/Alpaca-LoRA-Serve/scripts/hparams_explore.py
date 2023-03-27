import time
import itertools
import wandb
from transformers import GenerationConfig

wandb.login(key="")

PROJECT="txt_gen_test_project"

generation_configs = {
    "temperature": [0.5, 0.7, 0.8, 0.9, 1.0],
    "top_p": [0.5, 0.75, 0.85, 0.95, 1.0],
    "num_beams": [1, 2, 3, 4]
}

num_gens = 1

# token initialization
# model initialization

for comb in itertools.product(generation_configs['temperature'], 
                              generation_configs['top_p'],
                              generation_configs['num_beams']):
  temperature = comb[0]
  top_p = comb[1]
  num_beams = comb[2]

  generation_config = GenerationConfig(
      temperature=temperature,
      top_p=top_p,
      num_beams=num_beams,
  )

  first_columns = [f"gen_txt_{num}" for num in range(num_gens)]
  columns = first_columns + ["temperature", "top_p", "num_beams", "time_delta"]
  
  avg_time_delta = 0
  txt_gens = []
  for i in range(num_gens):
    start = time.time()
    # text generation
    text = "dummy text"
    txt_gens.append(text)

    # decode outputs
    end = time.time()
    t_delta = end - start
    avg_time_delta = avg_time_delta + t_delta

  avg_time_delta = round(avg_time_delta / num_gens, 4)
    
  wandb.init(
      project=PROJECT,
      name=f"t@{temperature}-tp@{top_p}-nb@{num_beams}",
      config=generation_config,
  )

  text_table = wandb.Table(columns=columns)
  text_table.add_data(*txt_gens, temperature, top_p, num_beams, avg_time_delta)

  wandb.log({
      "avg_t_delta": avg_time_delta,
      "results": text_table
  })

  wandb.finish()
