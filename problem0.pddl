(define (problem test_scenario)
  (:domain autonomous_vehicle_testing)
  (:objects
    ego_vehicle - vehicle
    agent1 agent2 - agent
  )
  (:init
    (vehicle ego_vehicle)
    (agent agent1)
    (agent agent2)
    (safe_distance ego_vehicle agent1)
    (safe_distance ego_vehicle agent2)
    (clear_path ego_vehicle)
    (speed_limit ego_vehicle)
    (lane_centered ego_vehicle)
    (ahead ego_vehicle agent1)
    (behind ego_vehicle agent2)
  )
  (:goal
    (and
      (safe_distance ego_vehicle agent1)
      (safe_distance ego_vehicle agent2)
      (clear_path ego_vehicle)
      (lane_centered ego_vehicle)
    )
  )
)