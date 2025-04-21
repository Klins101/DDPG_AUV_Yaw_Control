import numpy as np
import torch
import argparse
import os
import matplotlib.pyplot as plt
import control as ct
from everything_env import EverythingEnv
from ddpg_implementation import DDPG

def test_policy(policy, env, max_time=50.0):
    # Reset environment and initialize arrays for data collection
    state = env.reset()
    
    # Calculate max steps based on dt and max_time
    max_steps = int(max_time / env.dt)
    
    # Arrays to store data
    times = np.zeros(max_steps)
    outputs = np.zeros(max_steps)
    references = np.zeros(max_steps)
    errors = np.zeros(max_steps)
    actions = np.zeros(max_steps)
    actuator_actions = np.zeros(max_steps)  # Store actuator outputs
    uncertain_actions = np.zeros(max_steps)  # Store actions after uncertainty
    
    # Simulate system with trained policy
    for i in range(max_steps):
        # Select action from policy (no noise during testing)
        action = policy.select_action(state, add_noise=False)
        
        # Apply action and get new state
        next_state, _, done, info = env.step(action)
        
        # Store data
        times[i] = info['time']
        outputs[i] = info['output']
        references[i] = info['reference']
        errors[i] = state[1]  # error is the second element in state
        actions[i] = info['u_c']
        actuator_actions[i] = info['u_ac']
        uncertain_actions[i] = info['u_un']
        
        # Update state
        state = next_state
        
        if done:
            break
    
    # For simulations that end before max_time, truncate arrays
    if i < max_steps - 1:
        times = times[:i+1]
        outputs = outputs[:i+1]
        references = references[:i+1]
        errors = errors[:i+1]
        actions = actions[:i+1]
        actuator_actions = actuator_actions[:i+1]
        uncertain_actions = uncertain_actions[:i+1]
    
    # Calculate performance metrics
    e_ss = abs(errors[-20:].mean())  # Steady-state error (average of last 20 samples)
    ISE = np.sum(errors**2 * env.dt)  # Integral of Squared Error
    ITAE = np.sum(np.abs(errors) * times * env.dt)  # Integral of Time multiplied by Absolute Error
    IACE = np.sum(np.abs(actions) * env.dt)  # Integral of Absolute Control Effort
    
    # Calculate control rate
    action_rates = np.diff(actions) / env.dt
    IACER = np.sum(np.abs(action_rates) * env.dt)  # Integral of Absolute Control Effort Rate
    max_control = np.max(np.abs(actions))  # Maximum control value
    
    # Return all collected data and metrics
    return {
        'times': times,
        'outputs': outputs,
        'references': references,
        'errors': errors,
        'actions': actions,
        'actuator_actions': actuator_actions,
        'uncertain_actions': uncertain_actions,
        'metrics': {
            'e_ss': e_ss,
            'ISE': ISE,
            'ITAE': ITAE,
            'IACE': IACE,
            'IACER': IACER,
            'max_control': max_control
        }
    }

def save_ddpg_data(time, r, y, e, u_c, u_ac, filename='DDPG.txt'):
    """
    Stack time, reference, output, error, control command, and actuator output
    and save them in a text file with a fixed format.
    """
    data = np.vstack((time, r, y, e, u_c, u_ac)).T
    np.savetxt(filename, data, fmt='%.4f', delimiter=' ')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="models/best_ddpg_models", type=str)
    parser.add_argument("--reference", default=1.0, type=float)
    parser.add_argument("--reference-type", default="constant", type=str, 
                      choices=["constant", "sine", "step", "custom_step"],
                      help="Type of reference signal (constant, sine, step, custom_step)")
    parser.add_argument("--max-time", default=50.0, type=float)
    parser.add_argument("--disturbance", default=0.0, type=float)
    parser.add_argument("--disturbance-time", default=15.0, type=float)
    parser.add_argument("--noise", default=0.0, type=float)
    parser.add_argument("--noise-time", default=20.0, type=float)
    parser.add_argument("--use-actuator", default=False, type=bool)
    parser.add_argument("--use-uncertainty", default=True, type=bool)
    parser.add_argument("--gain", default=1.0, type=float)
    parser.add_argument("--delay", default=0.0, type=float)
    parser.add_argument("--sine-amplitude", default=40.0, type=float)
    parser.add_argument("--sine-freq", default=0.2, type=float)
    parser.add_argument("--step-value", default=1.0, type=float)
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists("Simulation Outcomes"):
        os.makedirs("Simulation Outcomes")
    
    # Set device and random seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    np.random.seed(0)
    
    # Create test environment with specified parameters
    env = EverythingEnv(
        reference_type=args.reference_type,
        reference=args.reference,
        disturbance_time=args.disturbance_time,
        disturbance_value=args.disturbance,
        noise_time=args.noise_time,
        noise_std=args.noise,
        use_actuator=args.use_actuator,
        use_uncertainty=args.use_uncertainty,
        gain=args.gain,
        delay=args.delay,
        sine_amplitude=args.sine_amplitude,
        sine_freq=args.sine_freq,
        step_value=args.step_value,
        TOTAL_TIME=args.max_time
    )
    
    # Get environment parameters
    state_dim = env.state_dim
    action_dim = env.action_dim
    max_action = env.max_action
    
    # Initialize DDPG agent
    agent = DDPG(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action
    )
    
    # Load trained model
    try:
        agent.load(args.model_path)
        print(f"Loaded model from {args.model_path}")
    except:
        print(f"Could not load model from {args.model_path}. Testing with untrained model.")
    
    # Test the policy
    results = test_policy(agent, env, max_time=args.max_time)
    
    # Extract results
    times = results['times']
    outputs = results['outputs']
    references = results['references']
    errors = results['errors']
    actions = results['actions']
    actuator_actions = results['actuator_actions']
    uncertain_actions = results['uncertain_actions']
    metrics = results['metrics']
    
    # Plot system response
    plt.figure(figsize=(12, 8))
    
    # Plot output and reference
    plt.subplot(2, 1, 1)
    plt.plot(times, outputs, 'b-', label='Output')
    plt.plot(times, references, 'r--', label='Reference')
    plt.xlabel('Time (s)')
    plt.ylabel('Yaw Angle')
    plt.legend()
    if args.reference_type == 'sine':
        plt.title(f'System Response - Sine Reference (Amp={args.sine_amplitude}, Freq={args.sine_freq})')
    elif args.reference_type == 'custom_step':
        plt.title('System Response - Custom Step Reference')
    else:
        plt.title(f'System Response - {args.reference_type.capitalize()} Reference')
    plt.grid(True)
    
    # Plot control actions
    plt.subplot(2, 1, 2)
    plt.plot(times, actions, 'y-', label='Control Command (u_c)')
    plt.plot(times, actuator_actions, 'b-', label='Actuator Output (u_ac)')
    plt.plot(times, uncertain_actions, 'r--', label='Uncertain Output (u_un)')
    plt.xlabel('Time (s)')
    plt.ylabel('Control Input')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Generate filename suffix based on reference type and parameters
    if args.reference_type == 'sine':
        filename_suffix = f"sine_amp{args.sine_amplitude}_freq{args.sine_freq}"
    elif args.reference_type == 'custom_step':
        filename_suffix = "custom_step"
    else:
        filename_suffix = f"{args.reference_type}_ref{args.reference}"
    
    # Add other parameters to filename
    if args.disturbance != 0.0:
        filename_suffix += f"_dist{args.disturbance}"
    if args.noise != 0.0:
        filename_suffix += f"_noise{args.noise}"
    if args.delay != 0.0:
        filename_suffix += f"_delay{args.delay}"
    if args.gain != 1.0:
        filename_suffix += f"_gain{args.gain}"
    
    # Save the plot
    plt.savefig(f"Simulation Outcomes/system_response_{filename_suffix}.png")
    
    # Save performance metrics
    with open(f"Simulation Outcomes/Criteria_ddpg_{filename_suffix}.txt", 'w') as f:
        f.write("Performance Metrics:\n")
        f.write(f"Steady-state error (e_ss): {metrics['e_ss']:.6f}\n")
        f.write(f"Integral of Squared Error (ISE): {metrics['ISE']:.6f}\n")
        f.write(f"Integral of Time multiplied by Absolute Error (ITAE): {metrics['ITAE']:.6f}\n")
        f.write(f"Integral of Absolute Control Effort (IACE): {metrics['IACE']:.6f}\n")
        f.write(f"Integral of Absolute Control Effort Rate (IACER): {metrics['IACER']:.6f}\n")
        f.write(f"Maximum control value: {metrics['max_control']:.6f}\n")
    
    # Save data in DDPG format
    save_ddpg_data(times, references, outputs, errors, actions, actuator_actions, 
                   filename=f"Simulation Outcomes/DDPG_{filename_suffix}.txt")
    
    # Print performance metrics
    print("\nPerformance Metrics:")
    print(f"Steady-state error (e_ss): {metrics['e_ss']:.6f}")
    print(f"Integral of Squared Error (ISE): {metrics['ISE']:.6f}")
    print(f"Integral of Time multiplied by Absolute Error (ITAE): {metrics['ITAE']:.6f}")
    print(f"Integral of Absolute Control Effort (IACE): {metrics['IACE']:.6f}")
    print(f"Integral of Absolute Control Effort Rate (IACER): {metrics['IACER']:.6f}")
    print(f"Maximum control value: {metrics['max_control']:.6f}")
    
    print(f"\nResults saved to:")
    print(f"  - Plot: Simulation Outcomes/system_response_{filename_suffix}.png")
    print(f"  - Metrics: Simulation Outcomes/Criteria_ddpg_{filename_suffix}.txt")
    print(f"  - Data: Simulation Outcomes/DDPG_{filename_suffix}.txt")

if __name__ == "__main__":
    main()