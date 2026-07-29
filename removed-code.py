#============================================================================
# Pre-fill Replay Memory with Random Experience
#============================================================================
def prefill_memory(length):
    print('Pre-filling replay memory...')
    obs, _ = env.reset()
    state = preprocess(obs)

    for i in range(length):
        # Take a fully random action (no network involved yet)
        action = torch.tensor([[random.randrange(num_actions)]], device=device, dtype=torch.long)
        a = action.item()

        obs, reward, done, truncated, info = env.step(a)
        reward_tensor = torch.tensor([reward], device=device)

        if done or truncated:
            next_state = None
        else:
            next_state = preprocess(obs)

        memory.push(state, action, next_state, reward_tensor)

        if done or truncated:
            obs, _ = env.reset()
            state = preprocess(obs)
        else:
            state = next_state

    print(f'Replay memory pre-filled with {len(memory)} experiences.')


# Warm-start the replay buffer before training begins
    prefill_memory(pretrain_length)



#============================================================================
# Evaluate Agent Performance
#============================================================================
print('Starting Evaluation...')
eval_counter = 0.0
total_steps = 0.0
total_reward = 0.0

for e in range(eval_episodes):
    # Initialise the Environment and State
    currentObs, _ = env.reset()
    currentState = preprocess(currentObs)

    # Main RL Loop
    for i in range(0, max_steps):
        # Always act greedily during evaluation (no exploration)
        action = select_action(currentState, greedy=True)
        a = action.item()

        obs, reward, done, truncated, info = env.step(a)

        if done or truncated:
            nextState = None
        else:
            nextState = preprocess(obs)

        if done or truncated:
            total_reward += reward
            total_steps += env.unwrapped.step_count
            if done:
                print('Finished evaluation episode %d with reward %f, %d steps, reaching goal '
                      % (e, reward, env.unwrapped.step_count))
                eval_counter += 1
            if truncated:
                print('Failed evaluation episode %d with reward %f, %d steps'
                      % (e, reward, env.unwrapped.step_count))
            break

        currentState = nextState

# Print a Summary of the Evaluation Results
print('Completion rate %.2f with average reward %0.4f and average steps %0.2f'
      % (eval_counter/eval_episodes, total_reward/eval_episodes, total_steps/eval_episodes))

writer.close()
    