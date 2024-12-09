create: docker run --shm-size=6g -p 5000:5000 --gpus all -it --name mamba_vision_container -v C:\My_Laptop\Repo\Palm-Print-Identification-System:/app vkev25811/mamba
start: docker start mamba_vision_container
start-terminal: docker exec -it mamba_vision_container bash



<!-- pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html -->