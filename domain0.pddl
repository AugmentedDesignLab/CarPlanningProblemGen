(define (domain driving_behavior)

  (:requirements :strips :typing)

  (:types vehicle environment)

  (:predicates
    (lane_position_maintained ?v - vehicle)
    (intersection_ahead ?v - vehicle)
    (traffic_light_green ?v - vehicle)
    (constant_speed ?v - vehicle)
    (velocity_maintained ?v - vehicle)
    (safe_environment ?v - vehicle)
    (no_pedestrians ?v - vehicle)
    (dry_surface ?e - environment)
    (driver_not_distracted ?v - vehicle)
    (surrounding_vehicle_detected ?v - vehicle)
    (relative_position_safe ?v - vehicle)
    (confidence_in_navigation ?v - vehicle)
    (green_light_stable ?v - vehicle)
  )

  (:action maintain_lane_position
    :parameters (?v - vehicle)
    :precondition (and (lane_position_maintained ?v)
                       (constant_speed ?v))
    :effect (lane_position_maintained ?v)
  )

  (:action approach_intersection
    :parameters (?v - vehicle)
    :precondition (and (intersection_ahead ?v)
                       (traffic_light_green ?v))
    :effect (velocity_maintained ?v)
  )

  (:action ensure_safe_environment
    :parameters (?v - vehicle ?e - environment)
    :precondition (and (no_pedestrians ?v)
                       (dry_surface ?e))
    :effect (driver_not_distracted ?v)
  )

  (:action monitor_surrounding_vehicles
    :parameters (?v - vehicle)
    :precondition (surrounding_vehicle_detected ?v)
    :effect (relative_position_safe ?v)
  )

  (:action navigate_intersection_safely
    :parameters (?v - vehicle)
    :precondition (green_light_stable ?v)
    :effect (confidence_in_navigation ?v)
  )
)