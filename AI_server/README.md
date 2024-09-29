create: docker run --gpus all -it --name mamba_vision_container -v C:\My_Laptop\Repo\Palm-Print-Identification-System:/app mamba_vision
start: docker start mamba_vision_container
start-terminal: docker exec -it mamba_vision_container bash
