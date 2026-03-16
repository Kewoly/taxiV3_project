from env import SimpleTaxiEnv


def test_rewards():
    """Teste directement les récompenses."""
    env = SimpleTaxiEnv()
    
    print("\n" + "="*60)
    print("TEST DIRECT DES RÉCOMPENSES")
    print("="*60)
    
    obs, _ = env.reset()
    print(f"\nÉtat initial :")
    print(f"  Taxi : ({env.taxi_x}, {env.taxi_y})")
    print(f"  Passager 1 : ({env.passengers[0]['x']}, {env.passengers[0]['y']}) → ({env.passengers[0]['dest_x']}, {env.passengers[0]['dest_y']})")
    print(f"  Passager 2 : ({env.passengers[1]['x']}, {env.passengers[1]['y']}) → ({env.passengers[1]['dest_x']}, {env.passengers[1]['dest_y']})")
    
    # Test 1 : Mouvement
    obs, reward, _, _, _ = env.step(0)
    print(f"\n1. Mouvement (Sud) : reward = {reward} (attendu : -1.0)")
    
    # Test 2 : Aller au premier pickup (1, 1)
    print(f"\n2. Navigation vers premier pickup (1, 1)...")
    for i in range(10):
        obs, reward, _, _, _ = env.step(3)  # Ouest
        if env.taxi_x == 1:
            break
    
    for i in range(10):
        obs, reward, _, _, _ = env.step(1)  # Nord
        if env.taxi_y == 1:
            break
    
    print(f"  Taxi maintenant à : ({env.taxi_x}, {env.taxi_y})")
    
    # Test 3 : Pickup
    obs, reward, _, _, _ = env.step(4)
    print(f"\n3. Pickup : reward = {reward} (attendu : 5.0)")
    print(f"  Passager 1 in_taxi : {env.passengers[0]['in_taxi']}")
    
    # Test 4 : Aller au premier dropoff (8, 1)
    print(f"\n4. Navigation vers premier dropoff (8, 1)...")
    for i in range(10):
        obs, reward, _, _, _ = env.step(2)  # Est
        if env.taxi_x == 8:
            break
    
    print(f"  Taxi maintenant à : ({env.taxi_x}, {env.taxi_y})")
    
    # Test 5 : Dropoff
    obs, reward, terminated, _, _ = env.step(5)
    print(f"\n5. Dropoff : reward = {reward} (attendu : 20.0)")
    print(f"  Passager 1 in_taxi : {env.passengers[0]['in_taxi']}")
    print(f"  Delivered : {env.delivered}")
    
    env.close()
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    test_rewards()
