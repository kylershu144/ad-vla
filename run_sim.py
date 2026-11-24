import carla
import random

client = carla.Client('192.168.4.38', 2000)
client.set_timeout(10.0)
world = client.load_world('Town10HD_Opt')

actors = []
camera = None
try:
    spawn_points = world.get_map().get_spawn_points()
    ego_spawn = spawn_points.pop(random.randrange(len(spawn_points)))

    ego_bp = world.get_blueprint_library().find('vehicle.lincoln.mkz')
    ego_bp.set_attribute('role_name', 'hero')
    ego_vehicle = world.spawn_actor(ego_bp, ego_spawn)
    actors.append(ego_vehicle)

    vehicle_blueprints = world.get_blueprint_library().filter('vehicle.*')
    for spawn_point in random.sample(spawn_points, 10):
        npc = world.try_spawn_actor(random.choice(vehicle_blueprints), spawn_point)
        if npc is not None:
            actors.append(npc)

    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera = world.spawn_actor(camera_bp, carla.Transform(carla.Location(z=1.5)), attach_to=ego_vehicle)
    camera.listen(lambda image: image.save_to_disk('out/%06d.png' % image.frame))

    for vehicle in world.get_actors().filter('vehicle.*'):
        vehicle.set_autopilot(True)

    input('Recording. Press enter to stop.\n')
finally:
    if camera is not None:
        camera.stop()
        camera.destroy()
    client.apply_batch([carla.command.DestroyActor(actor) for actor in actors])