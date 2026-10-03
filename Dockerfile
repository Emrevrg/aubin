# AUBIN server (GPU). Build:  docker build -t aubin .
# Run:   docker run --gpus all -p 8009:8009 -e AUBIN_MODEL=emrevrg/AUBIN-12B aubin
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
WORKDIR /app
RUN pip install --no-cache-dir -U "transformers>=4.50" accelerate bitsandbytes peft huggingface_hub
COPY aubin/ /app/aubin/
ENV AUBIN_MODEL=emrevrg/AUBIN-12B PORT=8009
EXPOSE 8009
CMD ["sh", "-c", "python -m aubin.cli serve --model $AUBIN_MODEL --port $PORT"]
