(define (problem driving-scenario)
  (:domain driving-behaviors)
  (:objects
    car1 - vehicle
    trafficLight1 - traffic-light
    pedestrian1 - pedestrian
    wetRoad - road-condition
    foggyCondition - visibility-condition
    distance5 distance10 - distance
  )
  (:init
    (red-light-ahead car1 trafficLight1)
    (within-distance car1 distance5)
    (pedestrian-crossing car1 pedestrian1)
    (within-distance car1 distance10)
    (wet-road wetRoad)
    (foggy-visibility foggyCondition)
  )
  (:goal
    (and
      (velocity-zero car1)
      (negative-acceleration car1)
      (speed-reduced car1 20)
    )
  )
)