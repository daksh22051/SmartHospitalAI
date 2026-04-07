"""
Data Correctness Test Script for Smart Hospital Orchestration
Run this to verify all data fields are correct
"""

import requests
import json

BASE_URL = "http://localhost:5000"

def test_initialization():
    """Test 1: Environment initialization"""
    response = requests.post(f"{BASE_URL}/api/init", json={'task': 'medium', 'seed': 42})
    data = response.json()
    
    checks = {
        'success': data.get('success') == True,
        'has_task': 'task' in data,
        'has_initial_state': 'initial_state' in data,
        'task_value': data.get('task') == 'medium'
    }
    
    print("TEST 1: Initialization")
    for check, result in checks.items():
        print(f"  {check}: {'✅ PASS' if result else '❌ FAIL'}")
    
    return all(checks.values())

def test_state_fields():
    """Test 2: All state fields present and correct types"""
    response = requests.get(f"{BASE_URL}/api/get_state")
    data = response.json()
    
    if not data.get('success'):
        print("TEST 2: State - ❌ FAIL (API error)")
        return False
    
    state = data.get('state', {})
    
    required_fields = {
        'total_patients': int,
        'waiting': int,
        'admitted': int,
        'available_doctors': int,
        'available_beds': int,
        'system_load': (int, float),
        'step': int
    }
    
    print("\nTEST 2: State Fields")
    all_pass = True
    for field, expected_type in required_fields.items():
        value = state.get(field)
        is_defined = value is not None
        is_correct_type = isinstance(value, expected_type) if isinstance(expected_type, type) else isinstance(value, expected_type)
        
        status = '✅' if is_defined and is_correct_type else '❌'
        print(f"  {field}: {value} {status}")
        
        if not (is_defined and is_correct_type):
            all_pass = False
    
    return all_pass

def test_math_consistency():
    """Test 3: Mathematical consistency"""
    response = requests.get(f"{BASE_URL}/api/get_state")
    data = response.json()
    
    if not data.get('success'):
        return False
    
    state = data.get('state', {})
    total = state.get('total_patients', 0)
    waiting = state.get('waiting', 0)
    admitted = state.get('admitted', 0)
    
    # Verify: total = waiting + admitted
    math_check = total == waiting + admitted
    
    print("\nTEST 3: Math Consistency")
    print(f"  {waiting} (waiting) + {admitted} (admitted) = {waiting + admitted}")
    print(f"  Expected total: {total}")
    print(f"  Match: {'✅ PASS' if math_check else '❌ FAIL'}")
    
    return math_check

def test_no_negative_values():
    """Test 4: No negative counts"""
    response = requests.get(f"{BASE_URL}/api/get_state")
    data = response.json()
    
    if not data.get('success'):
        return False
    
    state = data.get('state', {})
    count_fields = ['total_patients', 'waiting', 'admitted', 'available_doctors', 'available_beds']
    
    print("\nTEST 4: No Negative Values")
    all_positive = True
    for field in count_fields:
        value = state.get(field, 0)
        is_positive = value >= 0
        status = '✅' if is_positive else '❌'
        print(f"  {field}: {value} {status}")
        if not is_positive:
            all_positive = False
    
    return all_positive

def test_step_data():
    """Test 5: Step execution returns correct data"""
    response = requests.post(f"{BASE_URL}/api/step", json={'action': 1})
    data = response.json()
    
    print("\nTEST 5: Step Data")
    
    if not data.get('success'):
        print("  ❌ FAIL - Step execution failed")
        return False
    
    step = data.get('step', {})
    checks = {
        'has_action': 'action' in step,
        'has_reward': 'reward' in step,
        'has_info': 'info' in step,
        'reward_is_number': isinstance(step.get('reward'), (int, float)),
        'info_has_patients': 'num_patients' in step.get('info', {})
    }
    
    for check, result in checks.items():
        print(f"  {check}: {'✅ PASS' if result else '❌ FAIL'}")
    
    return all(checks.values())

def run_all_tests():
    """Run all data correctness tests"""
    print("="*60)
    print("SMART HOSPITAL DATA CORRECTNESS TEST SUITE")
    print("="*60)
    
    tests = [
        test_initialization,
        test_state_fields,
        test_math_consistency,
        test_no_negative_values,
        test_step_data
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n❌ ERROR in {test.__name__}: {e}")
            results.append(False)
    
    print("\n" + "="*60)
    print(f"RESULT: {sum(results)}/{len(results)} tests passed")
    if all(results):
        print("✅ ALL DATA CORRECT!")
    else:
        print("❌ SOME TESTS FAILED - Check output above")
    print("="*60)
    
    return all(results)

if __name__ == "__main__":
    run_all_tests()
