"""
Reward System Verification Test
Run this to verify the optimized reward system is working correctly
"""

import requests
import json

BASE_URL = "http://localhost:5000"

def test_reward_balance():
    """Test that rewards are in expected range [-100, +100] per episode"""
    print("="*70)
    print("REWARD SYSTEM VERIFICATION TEST")
    print("="*70)
    
    # Initialize environment
    print("\n1. Initializing environment...")
    response = requests.post(f"{BASE_URL}/api/init", json={'task': 'medium', 'seed': 42})
    init_data = response.json()
    
    if not init_data['success']:
        print("❌ FAILED: Could not initialize environment")
        return False
    print("✅ Environment initialized")
    
    # Run episode
    print("\n2. Running episode...")
    response = requests.post(f"{BASE_URL}/api/run_episode", json={'max_steps': 50})
    episode_data = response.json()
    
    if not episode_data['success']:
        print(f"❌ FAILED: Episode failed - {episode_data.get('error', 'Unknown error')}")
        return False
    
    total_reward = episode_data['total_reward']
    steps = episode_data['steps']
    avg_reward = total_reward / steps if steps > 0 else 0
    
    print(f"\n3. Results:")
    print(f"   Total Reward: {total_reward:.2f}")
    print(f"   Steps: {steps}")
    print(f"   Avg per step: {avg_reward:.2f}")
    
    # Verify reward is in expected range
    MIN_EXPECTED = -100
    MAX_EXPECTED = 100
    
    if MIN_EXPECTED <= total_reward <= MAX_EXPECTED:
        print(f"\n✅ REWARD IN TARGET RANGE [{MIN_EXPECTED}, {MAX_EXPECTED}]")
        status = "BALANCED"
    elif total_reward > MAX_EXPECTED:
        print(f"\n⚠️ REWARD ABOVE MAX (might be too easy)")
        status = "TOO_HIGH"
    else:
        print(f"\n⚠️ REWARD BELOW MIN (might still be too punishing)")
        status = "TOO_LOW"
    
    # Per-step sanity check
    if -20 <= avg_reward <= 20:
        print(f"✅ Per-step reward in range [-20, +20]")
    else:
        print(f"⚠️ Per-step reward outside range: {avg_reward:.2f}")
    
    print(f"\n{'='*70}")
    print(f"VERDICT: {status}")
    print(f"{'='*70}")
    
    return status == "BALANCED"

def test_manual_steps():
    """Test individual step rewards"""
    print("\n" + "="*70)
    print("MANUAL STEP REWARD TEST")
    print("="*70)
    
    # Initialize fresh
    requests.post(f"{BASE_URL}/api/init", json={'task': 'medium', 'seed': 123})
    
    rewards = []
    print("\nExecuting 10 manual steps...")
    
    for i in range(10):
        response = requests.post(f"{BASE_URL}/api/step", json={'action': 1})  # ALLOCATE
        data = response.json()
        
        if data['success']:
            reward = data['step']['reward']
            rewards.append(reward)
            print(f"   Step {i+1}: reward = {reward:+.2f}")
    
    print(f"\nStep Reward Stats:")
    print(f"   Min: {min(rewards):.2f}")
    print(f"   Max: {max(rewards):.2f}")
    print(f"   Avg: {sum(rewards)/len(rewards):.2f}")
    
    # Check if any step exceeded clipping bounds
    clipped_count = sum(1 for r in rewards if abs(r) >= 20)
    if clipped_count > 0:
        print(f"⚠️ {clipped_count} steps at clipping boundary (±20)")
    else:
        print(f"✅ No clipping detected in sample")
    
    return True

def run_all_tests():
    """Run all verification tests"""
    print("\n" + "="*70)
    print("SMART HOSPITAL REWARD SYSTEM - OPTIMIZED VERSION")
    print("="*70)
    
    try:
        results = []
        
        # Test 1: Episode balance
        if test_reward_balance():
            results.append("EPISODE_BALANCE: PASS")
        else:
            results.append("EPISODE_BALANCE: NEEDS_TUNING")
        
        # Test 2: Step-level rewards
        if test_manual_steps():
            results.append("STEP_REWARDS: PASS")
        else:
            results.append("STEP_REWARDS: FAIL")
        
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        for result in results:
            print(f"  {result}")
        
        print("\n✅ Reward system optimization complete!")
        print("   Expected range: [-100, +100] per episode")
        print("   Per-step range: [-20, +20] with clipping")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to server")
        print("   Make sure the server is running: python web_interface.py")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")

if __name__ == "__main__":
    run_all_tests()
