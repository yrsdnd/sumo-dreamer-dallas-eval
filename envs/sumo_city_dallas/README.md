# Custom Dallas-area SUMO city (round-2 training)

User-built Dallas-area network used for round-2 ego training. Original
hand-edited in `netedit`; round-1 fixes + driver-diversity tuning
documented in `scripts/city_setup/README.md`.

## Files

| File | Purpose |
|---|---|
| `osm.sumocfg` | SUMO main config (loads net + 3 trip files + output.add.xml) |
| `osm.net.xml.gz` | Compiled network (1,728 drivable edges, 2,009 junctions, 15 traffic lights) |
| `osm.passenger.trips.xml` | 1,606 passenger trips + `passengerDist` vTypeDistribution (97% normal + 3% dangerous) |
| `osm.bus.trips.xml` | 454 bus trips + 33 flows + `busDist` vTypeDistribution (real 12m buses on trips, default cars on flows) |
| `osm.motorcycle.trips.xml` | 1,350 motorcycle trips + `motorcycleDist` (95% normal + 5% dangerous) |
| `output.add.xml` | edgeData output config |
| `osm.netccfg` | netconvert config (used to build the net from OSM) |
| `osm.net.xml.gz`, `osm_bbox.osm.xml.gz` | the OSM-imported / compiled network |
| `City.netecfg` | netedit project file |
| `edgeData.xml` | edge-statistics output (auto-regenerated each run) |

## Quick run

```bash
cd envs/sumo_city_dallas
sumo -c osm.sumocfg
# or with GUI:
sumo-gui -c osm.sumocfg
```

## Stats (full 3,600 s sim)

- **5,896** vehicles loaded / inserted (5,542 user-defined + 354 round-1 dead-edge coverage)
- **0** teleports
- **25** collisions (all between non-ego environment vehicles, mostly at the E4 roundabout)
- **0** emergency stops, 22 emergency-braking events
- **5,395** trips completed in window
- Wait p50=12s, p95=98s, p99=218s, max=503s

See `scripts/city_setup/README.md` in the repo root for the full audit + fix history.
