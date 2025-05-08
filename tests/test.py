import torch
import time
import pygame
import numpy as np

def run_model_test(model_path, model_class, env, device='cuda' if torch.cuda.is_available() else 'cpu', render_fps=5):
    """
    Run a trained PyTorch model and display its actions using Pygame.

    Args:
        model_path (str): Path to the saved model weights (.pt or .pth).
        model_class (nn.Module): Your model class (not an instance).
        env: Your game environment (with reset, step, and pygame render method).
        device (str): 'cuda' or 'cpu'.
        render_fps (int): Frames per second for Pygame rendering.
    """
    # Load model
    model = model_class(env.config)  # assuming env.config holds the config
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    obs = env.reset()
    done = False

    clock = pygame.time.Clock()

    while not done:
        # Preprocess observation
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            hidden = model.representation(obs_tensor)
            policy = model.policy(hidden)
            action = torch.argmax(policy, dim=1).item()

        # Step in environment
        obs, reward, done, info = env.step(action)

        # Render in pygame
        env.render()  # assumes your env has this method
        pygame.display.flip()

        clock.tick(render_fps)

    print("Episode done. Final reward:", reward)

if __name__ == "__main__":
    run_model_test()