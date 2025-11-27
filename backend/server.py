import torch
import torch.nn.functional as F
import uvicorn
import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import sys
import os
from contextlib import asynccontextmanager # Added this import

# Add project root to sys.path to allow importing from model.py and sample.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Diffusion Model components
from model import DiffusionTransformer, DiffusionConfig, decode_tokens, encode_text
from sample import get_random_context # Helper for context tokens

# Import our projection utility
from backend.projection import RandomProjection

# Global variables for model and projection (loaded once)
model = None
projection_3d = None
dataset_tokens = None
device = None
config = DiffusionConfig()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, projection_3d, dataset_tokens, device, config

    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load model
    checkpoint_path = "weights/diffusion_model.pt"
    print(f"Loading model from {checkpoint_path}...")
    model = DiffusionTransformer(config).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print("Model loaded!")

    # Load dataset for random context sampling (if context_len > 0)
    if config.context_len > 0:
        print("Loading dataset for context sampling...")
        with open("data/tiny_shakespeare.txt", "r", encoding="utf-8") as f:
            text = f.read()
        dataset_tokens = encode_text(text)
        print(f"Loaded {len(dataset_tokens)} tokens from dataset")

    # Initialize Random Projection for 3D visualization
    # The `n_embd` from DiffusionConfig is the input_dim for projection
    projection_3d = RandomProjection(input_dim=config.n_embd, output_dim=3)
    print(f"Random Projection initialized from {config.n_embd}D to 3D.")

    yield # This indicates the startup is complete and the app can serve requests

    # Cleanup (optional, if there were shutdown tasks)
    print("Shutting down backend server.")


app = FastAPI(lifespan=lifespan) # Pass the lifespan function to FastAPI

@app.websocket("/ws/visualize")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            if data["action"] == "start_generation":
                prompt = data.get("prompt", "")
                seq_len = data.get("seq_len", config.sequence_len)
                num_steps = data.get("num_steps", config.sequence_len)
                temperature = data.get("temperature", 1.0)
                method = data.get("method", "confidence")
                confidence_threshold = data.get("confidence_threshold", 0.95)
                k = data.get("k", max(1, seq_len // 10))

                # Prepare context tokens
                context_tokens_batch = None
                if config.context_len > 0 and prompt:
                    # Use provided prompt as context
                    encoded_prompt = encode_text(prompt)
                    if len(encoded_prompt) > config.context_len:
                        encoded_prompt = encoded_prompt[-config.context_len:] # Truncate if too long
                    # Pad if too short, or just use as is if it fits
                    context_tokens_batch = torch.full(
                        (1, config.context_len),
                        config.mask_token_id,
                        dtype=torch.long,
                        device=device,
                    )
                    context_tokens_batch[0, :len(encoded_prompt)] = encoded_prompt.to(device)
                elif config.context_len > 0 and dataset_tokens is not None:
                     # Fallback to random context if no prompt and dataset available
                    context_tokens_batch = get_random_context(dataset_tokens, config.context_len, batch_size=1)

                print(f"Starting generation with seq_len={seq_len}, num_steps={num_steps}, method={method}")
                await stream_generation(
                    websocket,
                    model,
                    projection_3d,
                    seq_len,
                    num_steps,
                    temperature,
                    method,
                    confidence_threshold,
                    k,
                    device,
                    context_tokens_batch,
                )
            else:
                print(f"Unknown action: {data['action']}")

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")

async def stream_generation(
    websocket: WebSocket,
    model: DiffusionTransformer,
    projection_3d: RandomProjection,
    seq_len: int,
    num_steps: int,
    temperature: float,
    method: str,
    confidence_threshold: float,
    k: int,
    device: torch.device,
    context_tokens: torch.Tensor = None,
):
    batch_size = 1 # Always generating one sequence for visualization

    # Start from all mask tokens
    x = torch.full(
        (batch_size, seq_len),
        model.config.mask_token_id,
        dtype=torch.long,
        device=device,
    )

    # If context tokens provided, set them in the first context_len positions
    masked_positions = torch.ones(
        batch_size, seq_len, dtype=torch.bool, device=device
    )
    if context_tokens is not None:
        context_len = context_tokens.size(1)
        x[:, :context_len] = context_tokens
        masked_positions[:, :context_len] = False # Context tokens are not masked

    current_step = 0
    while current_step < num_steps and masked_positions.any():
        # Create timestep (use current_step as proxy for timestep)
        t_batch = torch.full((batch_size,), current_step, device=device, dtype=torch.long)
        t_batch = torch.clamp(t_batch, 0, model.config.diffusion_steps - 1)

        with torch.no_grad():
            # Predict tokens and get hidden states
            logits, hidden_states = model(x, t_batch, return_hidden_states=True)

            # Get confidence scores (max probability for each position)
            probs = F.softmax(logits / temperature, dim=-1)
            confidences, predicted_tokens = torch.max(probs, dim=-1)  # (B, T)

            # Project hidden states to 3D
            # hidden_states shape: (B, T, n_embd) -> (T, n_embd) for current sample
            projected_coords = projection_3d.project(hidden_states.squeeze(0)).cpu().numpy() # (T, 3)

            # --- Apply decoding logic based on method ---
            if method == "topk":
                # Mask out already-decoded positions for topk selection
                confidences_for_selection = confidences.masked_fill(~masked_positions, -float("inf"))
                k_actual = min(k, masked_positions.sum(dim=1).max().item())
                _, topk_indices = torch.topk(confidences_for_selection, k=k_actual, dim=1)  # (B, K)

                # Update the top-K positions
                for b in range(batch_size):
                    for idx_tensor in topk_indices[b]:
                        idx = idx_tensor.item()
                        if masked_positions[b, idx]: # Double check if it's still masked
                            x[b, idx] = predicted_tokens[b, idx]
                            masked_positions[b, idx] = False

            elif method == "confidence":
                # Select positions above threshold (only among masked positions)
                above_threshold = (confidences >= confidence_threshold) & masked_positions

                # Ensure at least one token is decoded per batch if any remain masked
                for b in range(batch_size):
                    if masked_positions[b].any() and not above_threshold[b].any():
                        # Decode the highest confidence masked token
                        masked_confidences = confidences[b].clone()
                        masked_confidences[~masked_positions[b]] = -float("inf")
                        best_idx = torch.argmax(masked_confidences)
                        above_threshold[b, best_idx] = True

                # Update positions above threshold
                x = torch.where(above_threshold, predicted_tokens, x)
                masked_positions = masked_positions & ~above_threshold
            # --- End decoding logic ---

            # Prepare data for WebSocket
            current_tokens_list = x.squeeze(0).tolist() # (T,)
            confidences_list = confidences.squeeze(0).tolist() # (T,)

            tokens_data = []
            for i in range(seq_len):
                tokens_data.append({
                    "id": current_tokens_list[i],
                    "char": decode_tokens(torch.tensor([current_tokens_list[i]]))[0], # Decode single char
                    "conf": confidences_list[i],
                    "pos": projected_coords[i].tolist(),
                    "is_masked": masked_positions.squeeze(0)[i].item() # Send mask status
                })

            response_data = {
                "type": "step_update",
                "data": {
                    "step": current_step,
                    "max_steps": num_steps,
                    "tokens": tokens_data
                }
            }

            await websocket.send_json(response_data)
            await asyncio.sleep(0.05) # Small delay to avoid overwhelming client/CPU for visualization pace

        current_step += 1

    # Send a final message when generation is complete
    final_tokens_list = x.squeeze(0).tolist()
    final_text = decode_tokens(torch.tensor(final_tokens_list, device=device))
    await websocket.send_json({
        "type": "generation_complete",
        "data": {
            "final_text": final_text,
            "tokens": tokens_data # Send the last state
        }
    })


# To run this server: uvicorn backend.server:app --port 50000

if __name__ == "__main__":
    # Disable reload to avoid socket sharing issues on Windows
    uvicorn.run("backend.server:app", host="127.0.0.1", port=50000, reload=False)
