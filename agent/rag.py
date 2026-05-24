import builtins
import json
import os
import sys
from typing import List, Dict, Any

try:
    import torch
except ImportError:
    torch = None

# auto code,
os.environ['TRUST_REMOTE_CODE'] = 'true'

# local model_path does not exist, Hugging Face model ID load
DEFAULT_HF_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"

try:
    import numpy as np
except ImportError:
    print('[ERROR] numpy not installed, please run: pip install numpy')
    raise

try:
    from sentence_transformers import SentenceTransformer, models
except ImportError:
    print('[ERROR] sentence-transformers not installed, please run: pip install sentence-transformers')
    raise

class CustomEmbeddingFunction: 
    def __init__(self, model_path: str, matryoshka_dim: int = None, device: str = 'cuda:0', model_type: str = None):
        "initialize embedding model Args: model_path: local model path(direct config.json root directory); is empty or does not exist Hugging Face load default model. matryoshka_dim: Matryoshka (optional, embedding) device: device('cpu' 'cuda'), None auto model_type: model type('nomic' 'qwen3'), None auto detect"
        # local path(empty, None path does not exist Hugging Face)
        model_path_str = (model_path or "").strip()
        use_hf_fallback = not model_path_str or not os.path.exists(model_path_str)

        if use_hf_fallback:
            # : Hugging Face load default embedding model
            _effective_model = DEFAULT_HF_EMBEDDING_MODEL
            print(f"[INFO] local path invalid is empty ({model_path!r}), Hugging Face load default model: {_effective_model}")
            self.model_type = 'qwen3'
            if device is None:
                if torch is None:
                    raise ImportError('torch not installed, please run: pip install torch')
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                self.device = device
            print(f"[INFO] usedevice: {self.device}")
            print('[INFO] Hugging Face download and load model()...')
            self.model = SentenceTransformer(_effective_model, device=self.device)
            self.model.eval()
            if torch is not None:
                torch.set_grad_enabled(False)
                torch.manual_seed(42)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(42)
            self.matryoshka_dim = matryoshka_dim
            test_embedding = self.model.encode(["test"], convert_to_numpy=True)
            self.full_dim = test_embedding.shape[1]
            if self.matryoshka_dim is not None:
                self.dimension = min(self.matryoshka_dim, self.full_dim)
                print(f"[INFO] use Matryoshka: {self.dimension} (complete: {self.full_dim})")
            else:
                self.dimension = self.full_dim
                print(f"[INFO] use complete: {self.dimension}")
            print(f"[SUCCESS] model initialize complete (Hugging Face, type: {self.model_type},: {self.dimension})")
            return

        model_path = model_path_str
        print(f"[INFO] load embedding model: {model_path}")
        # === auto detect model type ===
        if model_type is None:
            # path config file model type
            model_path_lower = model_path.lower()
            if 'qwen3' in model_path_lower or 'qwen' in model_path_lower:
                self.model_type = 'qwen3'
            elif 'nomic' in model_path_lower:
                self.model_type = 'nomic'
            else:
                # check config file
                config_path = os.path.join(model_path, 'config.json')
                if os.path.exists(config_path):
                    try:
                        with open(config_path, 'r', encoding='utf-8') as f:
                            config = json.load(f)
                        # check model type
                        model_arch = config.get("architectures", [])
                        if any("Qwen" in arch or "qwen" in arch.lower() for arch in model_arch):
                            self.model_type = 'qwen3'
                        elif any("Nomic" in arch or "nomic" in arch.lower() for arch in model_arch):
                            self.model_type = 'nomic'
                        else:
                            # default use qwen3(default model)
                            print(f"[WARN] config file model type, default use qwen3")
                            self.model_type = 'qwen3'
                    except Exception as e:
                        print(f"[WARN] read config file failed: {e}, default use qwen3")
                        self.model_type = 'qwen3'
                else:
                    print(f"[WARN] not found config.json, default use qwen3")
                    self.model_type = 'qwen3'
        else:
            self.model_type = model_type.lower()
            if self.model_type not in ['nomic', 'qwen3']:
                raise ValueError(f"unsupported model type: {model_type}, supports type: 'nomic', 'qwen3'")
        
        print(f"[INFO] model type: {self.model_type}")
        
        # model type check file
        if self.model_type == 'nomic':
            # Nomic model Python file
            required_files = ['config.json', 'modeling_hf_nomic_bert.py', 'configuration_hf_nomic_bert.py']
            missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]
            if missing_files:
                print(f"[WARN] local model directory missing Python code file: {missing_files}")
                print(f"[WARN] trust_remote_code=True, file exists: {model_path}")
        else:
            # Qwen3 model config.json
            if not os.path.exists(os.path.join(model_path, 'config.json')):
                print(f"[WARN] not found config.json file")
        
        # device
        if device is None:
            if torch is None:
                raise ImportError('torch not installed, please run: pip install torch')
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        print(f"[INFO] usedevice: {self.device}")
        
        print('[INFO] use sentence-transformers load local model...')
        # local load, request network
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_DATASETS_OFFLINE'] = '1'

        # === [ ] model directory sys. path, Python model ===
        if model_path not in sys.path:
            sys.path.insert(0, model_path)
            print(f"[INFO] model directory Python path: {model_path}")
        
        # replace input, auto code()
        _original_input = builtins.input
        def _auto_confirm_input(prompt=""):
            if "custom code" in prompt.lower() or "trust_remote_code" in prompt.lower():
                print('[INFO] auto code(trust_remote_code=True)')
                return 'y'
            return _original_input(prompt)
        builtins.input = _auto_confirm_input
        
        try:
            # === load: manual Sentence Transformer ===
            print(f"[INFO] use sentence_transformers. models. Transformer directlocal model directory Encoder")
            
            # manual Transformer, local load
            word_embedding_model = models.Transformer(
                model_path,  # directly use directory
                model_args={
                    "trust_remote_code": True,
                    "local_files_only": True  # load
                }
            )
            # Pooling
            pooling_model = models.Pooling(
                word_embedding_model.get_word_embedding_dimension(),
                pooling_mode_mean_tokens=True
            )
            # Sentence Transformer
            self.model = SentenceTransformer(
                modules=[word_embedding_model, pooling_model],
                device=self.device
            )
            
            # : model mode,,
            print('[INFO] model mode')
            self.model.eval()
            if torch is None:
                raise ImportError('torch not installed, please run: pip install torch')
            torch.set_grad_enabled(False)
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)
            
            print('[INFO] model load success')
        except Exception as e:
            print(f"[ERROR] model load failed: {e}")
            print('[INFO]:')
            print('[INFO] 1. model file complete')
            print('[INFO] 2. install sentence-transformers: pip install sentence-transformers')
            if self.model_type == 'nomic':
                print('[INFO] 3. model directory config file (configuration_hf_nomic_bert.py/modeling_hf_nomic_bert.py)')
            print('[INFO] 4. model directory config.json file(pytorch_model. bin model. safetensors)')
            raise
        finally:
            # original input
            builtins.input = _original_input
            # clean sys. path(optional)
            if model_path in sys.path:
                sys.path.remove(model_path)
        
        # Matryoshka
        self.matryoshka_dim = matryoshka_dim
        
        # get model
        # model type use
        if self.model_type == 'nomic':
            test_text = "search_query: test"
        else:
            # Qwen3
            test_text = "test"
        
        test_embedding = self.model.encode([test_text], convert_to_numpy=True)
        self.full_dim = test_embedding.shape[1]
        
        # actually use
        if self.matryoshka_dim is not None:
            self.dimension = min(self.matryoshka_dim, self.full_dim)
            print(f"[INFO] use Matryoshka: {self.dimension} (complete: {self.full_dim})")
        else:
            self.dimension = self.full_dim
            print(f"[INFO] use complete: {self.dimension}")
        
        print(f"[SUCCESS] model initialize complete (type: {self.model_type},: {self.dimension})")
    
    def name(self) -> str:
        'return embedding name(ChromaDB interface)'
        return "CustomEmbeddingFunction"
    
    def __call__(self, input: List[str], query: bool = False) -> List[List[float]]:
        'list embedding Args: input: list(ChromaDB 0.4.16+ interface parameter input) Returns: embedding list'
        # use sentence-transformers encode generate embeddings
        if query:
            prompt_name="query"
        else:
            prompt_name="document"
        embeddings = self.model.encode(
            input,
            prompt_name=prompt_name,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2
            show_progress_bar=False
        )
        
        # Matryoshka:, N
        if self.matryoshka_dim is not None:
            embeddings = embeddings[:, :self.matryoshka_dim]
            # , L2 calculate
            norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norm + 1e-10)
        
        # convert list
        return embeddings.tolist()
    
    def embed_documents(self, input: List[str]) -> List[List[float]]:
        'Chroma compatible interface: embedding(ChromaDB 0.4.16+ parameter input)'
        # input list
        if isinstance(input, str):
            input = [input]
        
        # model type
        if self.model_type == 'nomic':
            # Nomic v1.5
            input = ['search_document:' + text for text in input]
        # Qwen3, directly use original
        
        return self.__call__(input)
    
    def embed_query(self, input: str) -> List[float]:
        'Chroma compatible interface: query embedding(ChromaDB 0.4.16+ parameter input)'
        # input list,
        if isinstance(input, list):
            # list, directly use
            if len(input) > 0:
                text = input[0] if isinstance(input[0], str) else str(input[0])
            else:
                text = ""
        else:
            text = str(input)
        
        # model type
        if self.model_type == 'nomic':
            # Nomic v1.5 query
            text = 'search_query:' + text
        # Qwen3, directly use original
        
        result = self.__call__([text], query=True)[0]
        return result
