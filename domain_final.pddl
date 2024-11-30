(define (domain driving-behaviors)
  (:requirements :strips :typing)
  (:types vehicle traffic-light pedestrian road-condition visibility-condition distance number)
  
  (:predicates
    (red-light-ahead ?v - vehicle ?t - traffic-light)
    (within-distance ?v - vehicle ?d - distance)
    (pedestrian-crossing ?v - vehicle ?p - pedestrian)
    (wet-road ?r - road-condition)
    (foggy-visibility ?vc - visibility-condition)
    (velocity-zero ?v - vehicle)
    (negative-acceleration ?v - vehicle)
    (speed-reduced ?v - vehicle ?percent - number)
    (maintain-following-distance ?v - vehicle ?time - number)
  )
  
  (:action stop-at-red-light
    :parameters (?v - vehicle ?t - traffic-light ?d - distance)
    :precondition (and (red-light-ahead ?v ?t) (within-distance ?v ?d))
    :effect (velocity-zero ?v)
  )
  
  (:action slow-down-for-pedestrian
    :parameters (?v - vehicle ?p - pedestrian ?d - distance)
    :precondition (and (pedestrian-crossing ?v ?p) (within-distance ?v ?d))
    :effect (negative-acceleration ?v)
  )
  
  (:action adjust-speed-for-wet-road
    :parameters (?v - vehicle ?r - road-condition)
    :precondition (wet-road ?r)
    :effect (speed-reduced ?v 20)
  )
  
  (:action maintain-safe-distance
    :parameters (?v - vehicle ?d - distance ?time - number)
    :precondition (within-distance ?v ?d)
    :effect (maintain-following-distance ?v ?time)
  )
  
  (:action proceed-with-caution
    :parameters (?v - vehicle ?vc - visibility-condition)
    :precondition (foggy-visibility ?vc)
    :effect (speed-reduced ?v 30)
  )
)