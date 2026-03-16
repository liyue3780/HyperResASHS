import huggingface_hub as hf
import yaml

# Configure the HTTP backend to use requests with custom settings
def hf_disable_ssl_verification():
    if hasattr(hf, 'configure_http_backend'):
        import requests
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)        
        def backend_factory_requests() -> requests.Session:
            session = requests.Session()
            session.verify = False
            return session
        hf.configure_http_backend(backend_factory=backend_factory_requests)
    elif hasattr(hf, 'set_client_factory'):
        import httpx
        hf.set_client_factory(lambda : httpx.Client(verify=False))
        
        
# Also configure PyTorch to ignore SSL verification warnings when downloading models
def torch_hub_disable_ssl_verification():
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
        

def hf_read_yaml(repo_id:str, filename:str):
    """Read a YAML file from the Hugging Face Hub."""
    f_local = hf.hf_hub_download(repo_id=repo_id, filename=filename)
    with open(f_local, 'r') as f:
        return yaml.safe_load(f)