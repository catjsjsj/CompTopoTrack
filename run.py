

import dotenv
import hydra
from omegaconf import DictConfig
import os




dotenv.load_dotenv(override=True)

for _proxy_key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    _proxy_val = os.environ.get(_proxy_key, "")
    if "<" in _proxy_val or ">" in _proxy_val:
        os.environ.pop(_proxy_key, None)



@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):
    

    
    
    
    from src.train import train
    from src.utils import utils

    
    
    
    
    
    
    utils.extras(config)

    
    
    if config.get("print_config"):
        utils.print_config(config, resolve=True)

    
    return train(config)


if __name__ == "__main__":
    main()
