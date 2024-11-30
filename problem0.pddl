(define (problem driving-scenario)
  (:domain driving-behaviors)
  (:objects
    car1 - vehicle
    trafficLight1 - traffic-light
    pedestrian1 - pedestrian
    wetRoad - road-condition
    foggyCondition - visibility-condition
  )
  (:init
    (red-light-ahead car1 trafficLight1)
    (within-distance car1 5)
    (pedestrian-crossing car1 pedestrian1)
    (within-distance car1 10)
    (wet-road wetRoad)
    (foggy-visibility foggyCondition)
  )
  (:goal
    (and
      (velocity-zero car1)
      (negative-acceleration car1)
      (speed-reduced car1 20)
      (speed-reduced car1 30)
    )
  )
)