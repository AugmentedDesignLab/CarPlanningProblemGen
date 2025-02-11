(define (domain autonomous_vehicle_testing)
  (:requirements :strips :typing)
  (:types vehicle agent - object)

  (:predicates
    (safe_distance ?v - vehicle ?a - agent)
    (clear_path ?v