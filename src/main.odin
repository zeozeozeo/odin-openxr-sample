package main

import intr "base:intrinsics"
import "base:runtime"
import "core:fmt"
import "core:image"
import "core:log"
import "core:math"
import "core:math/linalg"
import "core:os"
import "core:slice"
import "core:strings"

import shared "../thirdparty/no_gfx/examples/shared"
import gltf2 "../thirdparty/no_gfx/examples/shared/gltf2"
import gpu "../thirdparty/no_gfx/gpu"
import xr "../thirdparty/openxr"

import sdl "vendor:sdl3"
import vk "vendor:vulkan"

App_Name :: "Odin OpenXR Sponza Walker"
Frames_In_Flight :: 2
Sponza_Path :: "sponza/glTF/Sponza.gltf"
Sky_Cubemap_Path :: "sponza/sky_cubemap_512x512.png"
Teleport_Deadzone :: 0.35
Teleport_Min_Range :: 0.35
Teleport_Max_Range :: 3.0
Player_Eye_Height :: 1.65
Shadow_Size :: 2048
Fullbright_Debug :: false
Sun_Spin_Rate :: 0.05
Sun_Orbit_Radius :: 2.0
Sun_Orbit_Height :: 10.0
Sun_Target :: [3]f32{0.0, 1.2, 0.0}

Gpu_Context :: struct {
	frame_sem:        gpu.Semaphore,
	next_frame:       u64,
	window_size:      [2]i32,
	scene_vert:       gpu.Shader,
	scene_frag:       gpu.Shader,
	shadow_vert:      gpu.Shader,
	shadow_frag:      gpu.Shader,
	sky_vert:         gpu.Shader,
	sky_frag:         gpu.Shader,
	helper_vert:      gpu.Shader,
	helper_frag:      gpu.Shader,
	desc_pool:        gpu.Descriptor_Pool,
	sampler_id:       u32,
	sky_sampler:      u32,
	shadow_sampler:   u32,
	shadow_map_id:    u32,
	sky_texture_id:   u32,
	loaded_textures:  [dynamic]gpu.Owned_Texture,
	shadow_map:       gpu.Owned_Texture,
	light_projection: Light_Projection,
	sun_direction:    [3]f32,
	sun_position:     [3]f32,
	sun_angle:        f32,
	scene_bounds_min: [3]f32,
	scene_bounds_max: [3]f32,
	scene_bounds_ok:  bool,
	scene:            shared.Scene,
	gltf_data:        ^gltf2.Data,
	meshes:           [dynamic]Mesh_GPU,
	sky_vertices:     gpu.slice_t([4]f32),
	sky_indices:      gpu.slice_t(u32),
	helper_vertices:  gpu.slice_t([4]f32),
	helper_indices:   gpu.slice_t(u32),
	window_depth:     gpu.Owned_Texture,
	frame_arenas:     [Frames_In_Flight]gpu.Arena,
}

Xr_Context :: struct {
	instance:         xr.Instance,
	system_id:        xr.SystemId,
	session:          xr.Session,
	app_space:        xr.Space,
	swapchain:        xr.Swapchain,
	swapchain_images: []xr.SwapchainImageVulkanKHR,
	textures:         []gpu.Texture,
	views:            []xr.View,
	config_views:     []xr.ViewConfigurationView,
	projection_views: []xr.CompositionLayerProjectionView,
	color_format:     gpu.Texture_Format,
	depth_texture:    gpu.Owned_Texture,
	session_state:    xr.SessionState,
	session_running:  bool,
	render_ready:     bool,
	stage_space:      bool,
	tracking_offset:  [3]f32,
	setup_failed:     bool,
	should_exit:      bool,
	action_set:       xr.ActionSet,
	aim_pose_action:  xr.Action,
	left_stick:       xr.Action,
	hand_paths:       [2]xr.Path,
	aim_spaces:       [2]xr.Space,
	hand_active:      [2]bool,
	hand_poses:       [2]xr.Posef,
	left_stick_value: [2]f32,
}

Window_Input :: struct {
	keys:       #sparse[sdl.Scancode]bool,
	mouse_dx:   f32,
	mouse_dy:   f32,
	mouse_look: bool,
}

Walker :: struct {
	xr_origin:              [3]f32,
	window_pos:             [3]f32,
	window_yaw_pitch:       [2]f32,
	teleport_active:        bool,
	teleport_target:        [3]f32,
	teleport_player_target: [3]f32,
	teleport_stick_was_on:  bool,
}

Mesh_GPU :: struct {
	pos:      gpu.slice_t([4]f32),
	normals:  gpu.slice_t([4]f32),
	tangents: gpu.slice_t([4]f32),
	uvs:      gpu.slice_t([2]f32),
	indices:  gpu.slice_t(u32),
}

Scene_Draw_Data :: struct #all_or_none {
	positions:             rawptr,
	normals:               rawptr,
	tangents:              rawptr,
	uvs:                   rawptr,
	model_to_world:        [16]f32,
	model_to_world_normal: [16]f32,
	world_to_view:         [16]f32,
	view_to_proj:          [16]f32,
	light_origin:          [4]f32,
	light_right:           [4]f32,
	light_up:              [4]f32,
	light_forward:         [4]f32,
	light_extents:         [4]f32,
	camera_world_pos:      [4]f32,
}

Scene_Frag_Data :: struct #all_or_none {
	base_color_map:                 u32,
	base_color_map_sampler:         u32,
	metallic_roughness_map:         u32,
	metallic_roughness_map_sampler: u32,
	normal_map:                     u32,
	normal_map_sampler:             u32,
	shadow_map:                     u32,
	shadow_map_sampler:             u32,
	sun_direction:                  [4]f32,
	light_extents:                  [4]f32,
	render_mode:                    u32,
	output_srgb:                    u32,
	sky_texture:                    u32,
	sky_sampler:                    u32,
}

Shadow_Draw_Data :: struct #all_or_none {
	positions:      rawptr,
	model_to_world: [16]f32,
	light_origin:   [4]f32,
	light_right:    [4]f32,
	light_up:       [4]f32,
	light_forward:  [4]f32,
	light_extents:  [4]f32,
}

Light_Projection :: struct {
	origin:  [3]f32,
	right:   [3]f32,
	up:      [3]f32,
	forward: [3]f32,
	width:   f32,
	height:  f32,
	near:    f32,
	far:     f32,
}

Helper_Draw_Data :: struct #all_or_none {
	positions:      rawptr,
	model_to_world: [16]f32,
	world_to_view:  [16]f32,
	view_to_proj:   [16]f32,
	color:          [4]f32,
}

Sky_Draw_Data :: struct #all_or_none {
	positions:     rawptr,
	world_to_view: [16]f32,
	view_to_proj:  [16]f32,
	sun_direction: [4]f32,
}

Sky_Frag_Data :: struct #all_or_none {
	sky_texture: u32,
	sky_sampler: u32,
	output_srgb: u32,
}

main :: proc() {
	ok := sdl.Init({.VIDEO})
	ensure(ok, fmt.tprintf("SDL_Init(SDL_INIT_VIDEO) failed: %v", sdl.GetError()))
	defer sdl.Quit()

	console_logger := log.create_console_logger()
	defer log.destroy_console_logger(console_logger)
	context.logger = console_logger

	window := sdl.CreateWindow(App_Name, 1200, 720, {.VULKAN, .RESIZABLE})
	ensure(window != nil, fmt.tprintf("SDL_CreateWindow failed: %v", sdl.GetError()))
	defer sdl.DestroyWindow(window)

	xr_ctx := xr_create_instance()
	if xr_try_get_system(&xr_ctx) {
		add_openxr_vulkan_device_extensions(xr_ctx.instance, xr_ctx.system_id)
	} else {
		log.info("No OpenXR HMD system is available yet; running windowed.")
	}

	gpu_ctx := gpu_init_for_window(window)
	defer gpu_context_destroy(&gpu_ctx)
	defer xr_destroy(&xr_ctx)

	input: Window_Input
	walker := Walker {
		xr_origin        = {0.0, 0.0, 0.0},
		window_pos       = {-7.58, 1.19, 0.26},
		window_yaw_pitch = {math.PI / 2.0, math.RAD_PER_DEG * 21.0},
	}

	ts_freq := sdl.GetPerformanceFrequency()
	now_ts := sdl.GetPerformanceCounter()
	retry_frame: u64
	was_xr_frame_mode := false
	for !xr_ctx.should_exit {
		last_ts := now_ts
		now_ts = sdl.GetPerformanceCounter()
		delta_time := min(0.1, f32(f64(now_ts - last_ts) / f64(ts_freq)))
		update_sun(&gpu_ctx, delta_time)

		if !handle_sdl_events(window, &input) {
			xr_ctx.should_exit = true
		}

		if xr_ctx.instance != {} {
			poll_xr_events(&xr_ctx)
		}

		if !xr_ctx.render_ready && !xr_ctx.setup_failed && retry_frame == 0 {
			if xr_ctx.system_id == {} && xr_try_get_system(&xr_ctx) {
				log.info(
					"OpenXR HMD system became available; recreating Vulkan device with runtime extensions.",
				)
				gpu_context_destroy(&gpu_ctx)
				add_openxr_vulkan_device_extensions(xr_ctx.instance, xr_ctx.system_id)
				gpu_ctx = gpu_init_for_window(window)
			}

			if xr_ctx.system_id != {} {
				xr_ctx.render_ready = xr_try_create_rendering(&xr_ctx)
			}
		}
		retry_frame = (retry_frame + 1) % 60

		xr_frame_mode := xr_ctx.session_running && xr_ctx.render_ready
		if was_xr_frame_mode && !xr_frame_mode {
			gpu.wait_idle()
		}
		was_xr_frame_mode = xr_frame_mode

		if xr_frame_mode {
			frame_state := xr.FrameState {
				sType = .FRAME_STATE,
			}
			wait_result := xr.WaitFrame(
				xr_ctx.session,
				&xr.FrameWaitInfo{sType = .FRAME_WAIT_INFO},
				&frame_state,
			)
			if wait_result != .SUCCESS {
				log.warnf("xrWaitFrame failed (%v); returning to window rendering.", wait_result)
				gpu.wait_idle()
				xr_ctx.session_running = false
				continue
			}
			xr_check(
				xr.BeginFrame(xr_ctx.session, &xr.FrameBeginInfo{sType = .FRAME_BEGIN_INFO}),
				"xrBeginFrame",
			)

			if !bool(frame_state.shouldRender) {
				xr_end_frame_empty(xr_ctx.session, frame_state.predictedDisplayTime)
			} else {
				if gpu_ctx.next_frame > Frames_In_Flight {
					gpu.semaphore_wait(gpu_ctx.frame_sem, gpu_ctx.next_frame - Frames_In_Flight)
				}

				if render_xr_frame(window, &gpu_ctx, &xr_ctx, &walker, frame_state) {
					gpu_ctx.next_frame += 1
				}
			}
		} else {
			update_window_walker(&walker, &input, delta_time)
			render_window_frame(window, &gpu_ctx, &walker)
		}
	}

	gpu.wait_idle()
}

gpu_init_for_window :: proc(window: ^sdl.Window) -> Gpu_Context {
	ensure(gpu.init(), "gpu.init() failed")
	gpu.swapchain_init_from_sdl(window, Frames_In_Flight)

	res := Gpu_Context {
		frame_sem  = gpu.semaphore_create(0),
		next_frame = 1,
	}
	for &arena in res.frame_arenas do arena = gpu.arena_init()
	sdl.GetWindowSize(window, &res.window_size.x, &res.window_size.y)
	gpu_context_create_scene_resources(&res)
	gpu_context_resize_window_depth(&res)
	return res
}

gpu_context_destroy :: proc(ctx: ^Gpu_Context) {
	if ctx.frame_sem != {} {
		gpu.wait_idle()
		if ctx.window_depth.handle != nil do gpu.texture_free_and_destroy(&ctx.window_depth)
		for &mesh in ctx.meshes do mesh_destroy(&mesh)
		delete(ctx.meshes)
		if ctx.helper_vertices.gpu.ptr != nil do gpu.mem_free(ctx.helper_vertices)
		if ctx.helper_indices.gpu.ptr != nil do gpu.mem_free(ctx.helper_indices)
		if ctx.gltf_data != nil do gltf2.unload(ctx.gltf_data)
		shared.destroy_scene(&ctx.scene)
		for &arena in ctx.frame_arenas do gpu.arena_destroy(&arena)
		if ctx.scene_vert != nil do gpu.shader_destroy(ctx.scene_vert)
		if ctx.scene_frag != nil do gpu.shader_destroy(ctx.scene_frag)
		if ctx.shadow_vert != nil do gpu.shader_destroy(ctx.shadow_vert)
		if ctx.shadow_frag != nil do gpu.shader_destroy(ctx.shadow_frag)
		if ctx.sky_vert != nil do gpu.shader_destroy(ctx.sky_vert)
		if ctx.sky_frag != nil do gpu.shader_destroy(ctx.sky_frag)
		if ctx.helper_vert != nil do gpu.shader_destroy(ctx.helper_vert)
		if ctx.helper_frag != nil do gpu.shader_destroy(ctx.helper_frag)
		if ctx.sky_vertices.gpu.ptr != nil do gpu.mem_free(ctx.sky_vertices)
		if ctx.sky_indices.gpu.ptr != nil do gpu.mem_free(ctx.sky_indices)
		for &tex in ctx.loaded_textures do gpu.texture_free_and_destroy(&tex)
		delete(ctx.loaded_textures)
		if ctx.shadow_map.handle != nil do gpu.texture_free_and_destroy(&ctx.shadow_map)
		gpu.desc_pool_destroy(&ctx.desc_pool)
		gpu.semaphore_destroy(ctx.frame_sem)
		gpu.cleanup()
	}
	ctx^ = {}
}

gpu_context_create_scene_resources :: proc(ctx: ^Gpu_Context) {
	ctx.scene_vert = gpu.shader_create(#load("shaders/scene.vert.spv", []u32), .Vertex)
	ctx.scene_frag = gpu.shader_create(#load("shaders/scene.frag.spv", []u32), .Fragment)
	ctx.shadow_vert = gpu.shader_create(#load("shaders/shadow.vert.spv", []u32), .Vertex)
	ctx.shadow_frag = gpu.shader_create(#load("shaders/shadow.frag.spv", []u32), .Fragment)
	ctx.sky_vert = gpu.shader_create(#load("shaders/sky.vert.spv", []u32), .Vertex)
	ctx.sky_frag = gpu.shader_create(#load("shaders/sky.frag.spv", []u32), .Fragment)
	ctx.helper_vert = gpu.shader_create(#load("shaders/helper.vert.spv", []u32), .Vertex)
	ctx.helper_frag = gpu.shader_create(#load("shaders/helper.frag.spv", []u32), .Fragment)
	ctx.desc_pool = gpu.desc_pool_create()
	ctx.sampler_id = gpu.desc_pool_alloc_sampler(
		&ctx.desc_pool,
		gpu.sampler_descriptor({max_anisotropy = min(16.0, gpu.device_limits().max_anisotropy)}),
	)
	ctx.sky_sampler = gpu.desc_pool_alloc_sampler(
		&ctx.desc_pool,
		gpu.sampler_descriptor(
			{
				min_filter = .Nearest,
				mag_filter = .Nearest,
				mip_filter = .Nearest,
				address_mode_u = .Clamp_To_Edge,
				address_mode_v = .Clamp_To_Edge,
				address_mode_w = .Clamp_To_Edge,
			},
		),
	)
	ctx.shadow_sampler = gpu.desc_pool_alloc_sampler(
		&ctx.desc_pool,
		gpu.sampler_descriptor(
			{
				address_mode_u = .Clamp_To_Edge,
				address_mode_v = .Clamp_To_Edge,
				address_mode_w = .Clamp_To_Edge,
			},
		),
	)
	upload_arena := gpu.arena_init()
	defer gpu.arena_destroy(&upload_arena)

	texture_infos: []shared.Gltf_Texture_Info
	ctx.scene, texture_infos, ctx.gltf_data = load_sponza_scene(Sponza_Path)
	ctx.scene_bounds_min, ctx.scene_bounds_max, ctx.scene_bounds_ok = compute_scene_bounds(ctx.scene)
	update_sun(ctx, 0)

	upload_cmd_buf := gpu.commands_begin(.Main)
	default_base := create_solid_texture(&upload_arena, upload_cmd_buf, {210, 205, 190, 255})
	default_mr := create_solid_texture(&upload_arena, upload_cmd_buf, {0, 185, 0, 255})
	default_normal := create_solid_texture(&upload_arena, upload_cmd_buf, {128, 128, 255, 255})
	sky_texture := load_texture_file(Sky_Cubemap_Path, &upload_arena, upload_cmd_buf)
	append(&ctx.loaded_textures, default_base)
	append(&ctx.loaded_textures, default_mr)
	append(&ctx.loaded_textures, default_normal)
	append(&ctx.loaded_textures, sky_texture)
	default_base_id := gpu.desc_pool_alloc_texture(
		&ctx.desc_pool,
		gpu.texture_view_descriptor(default_base, {}),
	)
	default_mr_id := gpu.desc_pool_alloc_texture(
		&ctx.desc_pool,
		gpu.texture_view_descriptor(default_mr, {}),
	)
	default_normal_id := gpu.desc_pool_alloc_texture(
		&ctx.desc_pool,
		gpu.texture_view_descriptor(default_normal, {}),
	)
	ctx.sky_texture_id = gpu.desc_pool_alloc_texture(
		&ctx.desc_pool,
		gpu.texture_view_descriptor(sky_texture, {}),
	)
	for &mesh in ctx.scene.meshes {
		mesh.base_color_map = default_base_id
		mesh.metallic_roughness_map = default_mr_id
		mesh.normal_map = default_normal_id
	}
	load_scene_textures(ctx, texture_infos, &upload_arena, upload_cmd_buf)
	for mesh in ctx.scene.meshes {
		append(&ctx.meshes, upload_mesh(&upload_arena, upload_cmd_buf, mesh))
	}
	ctx.sky_vertices, ctx.sky_indices = create_sky_cube(&upload_arena, upload_cmd_buf)
	ctx.helper_vertices, ctx.helper_indices = create_helper_mesh(&upload_arena, upload_cmd_buf)
	ctx.shadow_map = gpu.texture_alloc_and_create(
		{
			type = .D2,
			dimensions = {Shadow_Size, Shadow_Size, 1},
			format = .D32_Float,
			usage = {.Depth_Stencil_Attachment, .Sampled},
		},
	)
	ctx.shadow_map_id = gpu.desc_pool_alloc_texture(
		&ctx.desc_pool,
		gpu.texture_view_descriptor(ctx.shadow_map, {}),
	)
	gpu.cmd_barrier(upload_cmd_buf, .Transfer, .All, {})
	gpu.queue_submit(.Main, {upload_cmd_buf})
	gpu.queue_wait_idle(.Main)
}

gpu_context_resize_window_depth :: proc(ctx: ^Gpu_Context) {
	if ctx.window_depth.handle != nil {
		gpu.queue_wait_idle(.Main)
		gpu.texture_free_and_destroy(&ctx.window_depth)
	}
	ctx.window_depth = gpu.texture_alloc_and_create(
		{
			dimensions = {u32(max(1, ctx.window_size.x)), u32(max(1, ctx.window_size.y)), 1},
			format = .D32_Float,
			usage = {.Depth_Stencil_Attachment},
		},
	)
}

load_sponza_scene :: proc(
	path: string,
) -> (
	shared.Scene,
	[]shared.Gltf_Texture_Info,
	^gltf2.Data,
) {
	data, err := gltf2.load_from_file(path)
	if err != nil {
		log.errorf("Failed to load Sponza glTF: %v", err)
		panic("Sponza glTF load failed")
	}

	meshes: [dynamic]shared.Mesh
	texture_infos: [dynamic]shared.Gltf_Texture_Info
	start_idx: [dynamic]u32
	defer delete(start_idx)

	for mesh, mesh_i in data.meshes {
		append(&start_idx, u32(len(meshes)))
		for primitive in mesh.primitives {
			assert(primitive.mode == .Triangles)

			positions := shared.buffer_slice_with_stride(
				[3]f32,
				data,
				primitive.attributes["POSITION"],
				context.temp_allocator,
			)
			normals := shared.buffer_slice_with_stride(
				[3]f32,
				data,
				primitive.attributes["NORMAL"],
				context.temp_allocator,
			)
			indices := gltf2.buffer_slice(data, primitive.indices.?)

			indices_u32: [dynamic]u32
			defer delete(indices_u32)
			#partial switch ids in indices {
			case []u16:
				for id in ids do append(&indices_u32, u32(id))
			case []u32:
				for id in ids do append(&indices_u32, id)
			case:
				assert(false, "Unsupported Sponza index type")
			}

			pos4 := shared.to_vec4_array(positions, context.temp_allocator)
			norm4 := shared.to_vec4_array(normals, context.temp_allocator)
			tangent4 := make([][4]f32, len(positions), allocator = context.temp_allocator)
			if tangent_accessor, ok := primitive.attributes["TANGENT"]; ok {
				tangent_src := shared.buffer_slice_with_stride(
					[4]f32,
					data,
					tangent_accessor,
					context.temp_allocator,
				)
				copy(tangent4, tangent_src)
			} else {
				for &t in tangent4 do t = {1.0, 0.0, 0.0, 1.0}
			}
			uvs := make([][2]f32, len(positions), allocator = context.temp_allocator)
			if uv_accessor, ok := primitive.attributes["TEXCOORD_0"]; ok {
				uv_src := shared.buffer_slice_with_stride(
					[2]f32,
					data,
					uv_accessor,
					context.temp_allocator,
				)
				copy(uvs, uv_src)
			}

			mesh_idx := u32(len(meshes))
			if primitive.material != nil {
				material := data.materials[primitive.material.?]
				if material.metallic_roughness != nil {
					mr := material.metallic_roughness.?
					if mr.base_color_texture != nil {
						append(
							&texture_infos,
							texture_info_for(
								data,
								mesh_idx,
								.Base_Color,
								mr.base_color_texture.?.index,
							),
						)
					}
					if mr.metallic_roughness_texture != nil {
						append(
							&texture_infos,
							texture_info_for(
								data,
								mesh_idx,
								.Metallic_Roughness,
								mr.metallic_roughness_texture.?.index,
							),
						)
					}
				}
				if material.normal_texture != nil {
					append(
						&texture_infos,
						texture_info_for(data, mesh_idx, .Normal, material.normal_texture.?.index),
					)
				}
			}

			append(
				&meshes,
				shared.Mesh {
					pos = slice.clone_to_dynamic(pos4),
					normals = slice.clone_to_dynamic(norm4),
					tangents = slice.clone_to_dynamic(tangent4),
					uvs = slice.clone_to_dynamic(uvs),
					indices = slice.clone_to_dynamic(indices_u32[:]),
				},
			)
		}
		_ = mesh_i
	}

	instances: [dynamic]shared.Instance
	flip_z: matrix[4, 4]f32 = 1
	flip_z[2, 2] = -1

	traverse_node :: proc(
		instances: ^[dynamic]shared.Instance,
		data: ^gltf2.Data,
		start_idx: []u32,
		parent_transform: matrix[4, 4]f32,
		node_idx: int,
		flip_z: matrix[4, 4]f32,
	) {
		node := data.nodes[node_idx]
		local_transform := shared.xform_to_mat(node.translation, node.rotation, node.scale)
		transform := parent_transform * local_transform

		if node.mesh != nil {
			mesh_idx := node.mesh.?
			for _, primitive_i in data.meshes[mesh_idx].primitives {
				append(
					instances,
					shared.Instance {
						transform = flip_z * transform,
						mesh_idx = start_idx[mesh_idx] + u32(primitive_i),
					},
				)
			}
		}

		for child in node.children {
			traverse_node(instances, data, start_idx, transform, int(child), flip_z)
		}
	}

	for node_idx in data.scenes[0].nodes {
		traverse_node(&instances, data, start_idx[:], 1, int(node_idx), flip_z)
	}

	log.infof(
		"Loaded Sponza from %s: %d mesh primitives, %d instances",
		path,
		len(meshes),
		len(instances),
	)
	return shared.Scene{meshes = meshes, instances = instances}, texture_infos[:], data
}

texture_info_for :: proc(
	data: ^gltf2.Data,
	mesh_id: u32,
	texture_type: shared.Texture_Type,
	texture_index: gltf2.Integer,
) -> shared.Gltf_Texture_Info {
	image_index := int(data.textures[texture_index].source.?)
	return {mesh_id = mesh_id, texture_type = texture_type, image_index = image_index}
}

create_solid_texture :: proc(
	upload_arena: ^gpu.Arena,
	cmd_buf: gpu.Command_Buffer,
	rgba: [4]u8,
) -> gpu.Owned_Texture {
	staging := gpu.arena_alloc(upload_arena, u8, 4)
	for c, i in rgba do staging.cpu[i] = c
	texture := gpu.texture_alloc_and_create(
		{dimensions = {1, 1, 1}, format = .RGBA8_Unorm, usage = {.Sampled}},
	)
	gpu.cmd_copy_to_texture(cmd_buf, texture, staging)
	return texture
}

load_scene_textures :: proc(
	ctx: ^Gpu_Context,
	texture_infos: []shared.Gltf_Texture_Info,
	upload_arena: ^gpu.Arena,
	cmd_buf: gpu.Command_Buffer,
) {
	image_to_descriptor: map[int]u32
	defer delete(image_to_descriptor)

	for info in texture_infos {
		texture_id: u32
		if existing, ok := image_to_descriptor[info.image_index]; ok {
			texture_id = existing
		} else {
			img := shared.load_texture_from_gltf(info.image_index, ctx.gltf_data)
			texture := upload_image_texture(img, upload_arena, cmd_buf)
			image.destroy(img)
			append(&ctx.loaded_textures, texture)
			texture_id = gpu.desc_pool_alloc_texture(
				&ctx.desc_pool,
				gpu.texture_view_descriptor(texture, {}),
			)
			image_to_descriptor[info.image_index] = texture_id
		}

		switch info.texture_type {
		case .Base_Color:
			ctx.scene.meshes[info.mesh_id].base_color_map = texture_id
		case .Metallic_Roughness:
			ctx.scene.meshes[info.mesh_id].metallic_roughness_map = texture_id
		case .Normal:
			ctx.scene.meshes[info.mesh_id].normal_map = texture_id
		}
	}

	log.infof("Loaded %d unique Sponza textures", len(image_to_descriptor))
}

upload_image_texture :: proc(
	img: ^image.Image,
	upload_arena: ^gpu.Arena,
	cmd_buf: gpu.Command_Buffer,
) -> gpu.Owned_Texture {
	staging := gpu.arena_alloc_raw(upload_arena, len(img.pixels.buf), 1, 16)
	runtime.mem_copy(staging.cpu, raw_data(img.pixels.buf), len(img.pixels.buf))

	texture := gpu.texture_alloc_and_create(
		{
			type = .D2,
			dimensions = {u32(img.width), u32(img.height), 1},
			format = .RGBA8_Unorm,
			usage = {.Sampled},
		},
	)
	gpu.cmd_copy_to_texture(cmd_buf, texture, staging)
	return texture
}

load_texture_file :: proc(
	path: string,
	upload_arena: ^gpu.Arena,
	cmd_buf: gpu.Command_Buffer,
) -> gpu.Owned_Texture {
	bytes, err := os.read_entire_file_from_path(path, context.allocator)
	ensure(err == nil, fmt.tprintf("Could not read texture file: %s", path))
	defer delete(bytes)

	img, img_err := image.load_from_bytes(bytes, {.alpha_add_if_missing})
	ensure(img_err == nil, fmt.tprintf("Could not decode texture file: %s", path))
	defer image.destroy(img)

	return upload_image_texture(img, upload_arena, cmd_buf)
}

upload_mesh :: proc(
	upload_arena: ^gpu.Arena,
	cmd_buf: gpu.Command_Buffer,
	mesh: shared.Mesh,
) -> Mesh_GPU {
	positions_staging := gpu.arena_alloc(upload_arena, [4]f32, len(mesh.pos))
	normals_staging := gpu.arena_alloc(upload_arena, [4]f32, len(mesh.normals))
	tangents_staging := gpu.arena_alloc(upload_arena, [4]f32, len(mesh.tangents))
	uvs_staging := gpu.arena_alloc(upload_arena, [2]f32, len(mesh.uvs))
	indices_staging := gpu.arena_alloc(upload_arena, u32, len(mesh.indices))
	copy(positions_staging.cpu, mesh.pos[:])
	copy(normals_staging.cpu, mesh.normals[:])
	copy(tangents_staging.cpu, mesh.tangents[:])
	copy(uvs_staging.cpu, mesh.uvs[:])
	copy(indices_staging.cpu, mesh.indices[:])

	res: Mesh_GPU
	res.pos = gpu.mem_alloc([4]f32, len(mesh.pos), gpu.Memory.GPU)
	res.normals = gpu.mem_alloc([4]f32, len(mesh.normals), gpu.Memory.GPU)
	res.tangents = gpu.mem_alloc([4]f32, len(mesh.tangents), gpu.Memory.GPU)
	res.uvs = gpu.mem_alloc([2]f32, len(mesh.uvs), gpu.Memory.GPU)
	res.indices = gpu.mem_alloc(u32, len(mesh.indices), gpu.Memory.GPU)
	gpu.cmd_mem_copy(cmd_buf, res.pos, positions_staging)
	gpu.cmd_mem_copy(cmd_buf, res.normals, normals_staging)
	gpu.cmd_mem_copy(cmd_buf, res.tangents, tangents_staging)
	gpu.cmd_mem_copy(cmd_buf, res.uvs, uvs_staging)
	gpu.cmd_mem_copy(cmd_buf, res.indices, indices_staging)
	return res
}

mesh_destroy :: proc(mesh: ^Mesh_GPU) {
	gpu.mem_free(mesh.pos)
	gpu.mem_free(mesh.normals)
	gpu.mem_free(mesh.tangents)
	gpu.mem_free(mesh.uvs)
	gpu.mem_free(mesh.indices)
	mesh^ = {}
}

create_helper_mesh :: proc(
	upload_arena: ^gpu.Arena,
	cmd_buf: gpu.Command_Buffer,
) -> (
	gpu.slice_t([4]f32),
	gpu.slice_t(u32),
) {
	verts := [?][4]f32 {
		{-0.035, 0.02, 0.0, 1},
		{0.035, 0.02, 0.0, 1},
		{0.035, 0.02, 1.0, 1},
		{-0.035, 0.02, 1.0, 1},
		{-0.35, 0.025, -0.35, 1},
		{0.35, 0.025, -0.35, 1},
		{0.35, 0.025, 0.35, 1},
		{-0.35, 0.025, 0.35, 1},
		{-0.08, -0.08, -0.08, 1},
		{0.08, -0.08, -0.08, 1},
		{0.08, 0.08, -0.08, 1},
		{-0.08, 0.08, -0.08, 1},
		{-0.08, -0.08, 0.08, 1},
		{0.08, -0.08, 0.08, 1},
		{0.08, 0.08, 0.08, 1},
		{-0.08, 0.08, 0.08, 1},
	}
	indices := [?]u32 {
		0,
		1,
		2,
		0,
		2,
		3,
		4,
		5,
		6,
		4,
		6,
		7,
		8,
		10,
		9,
		8,
		11,
		10,
		12,
		13,
		14,
		12,
		14,
		15,
		8,
		9,
		13,
		8,
		13,
		12,
		9,
		10,
		14,
		9,
		14,
		13,
		10,
		11,
		15,
		10,
		15,
		14,
		11,
		8,
		12,
		11,
		12,
		15,
	}

	verts_staging := gpu.arena_alloc(upload_arena, [4]f32, len(verts))
	indices_staging := gpu.arena_alloc(upload_arena, u32, len(indices))
	copy(verts_staging.cpu, verts[:])
	copy(indices_staging.cpu, indices[:])

	verts_gpu := gpu.mem_alloc([4]f32, len(verts), gpu.Memory.GPU)
	indices_gpu := gpu.mem_alloc(u32, len(indices), gpu.Memory.GPU)
	gpu.cmd_mem_copy(cmd_buf, verts_gpu, verts_staging)
	gpu.cmd_mem_copy(cmd_buf, indices_gpu, indices_staging)
	return verts_gpu, indices_gpu
}

create_sky_cube :: proc(
	upload_arena: ^gpu.Arena,
	cmd_buf: gpu.Command_Buffer,
) -> (
	gpu.slice_t([4]f32),
	gpu.slice_t(u32),
) {
	verts := [?][4]f32 {
		{-1, -1, -1, 1},
		{1, -1, -1, 1},
		{1, 1, -1, 1},
		{-1, 1, -1, 1},
		{-1, -1, 1, 1},
		{1, -1, 1, 1},
		{1, 1, 1, 1},
		{-1, 1, 1, 1},
	}
	indices := [?]u32 {
		0,
		2,
		1,
		0,
		3,
		2,
		5,
		6,
		4,
		6,
		7,
		4,
		4,
		7,
		0,
		7,
		3,
		0,
		1,
		2,
		5,
		2,
		6,
		5,
		3,
		7,
		2,
		7,
		6,
		2,
		4,
		0,
		5,
		0,
		1,
		5,
	}
	verts_staging := gpu.arena_alloc(upload_arena, [4]f32, len(verts))
	indices_staging := gpu.arena_alloc(upload_arena, u32, len(indices))
	copy(verts_staging.cpu, verts[:])
	copy(indices_staging.cpu, indices[:])
	verts_gpu := gpu.mem_alloc([4]f32, len(verts), gpu.Memory.GPU)
	indices_gpu := gpu.mem_alloc(u32, len(indices), gpu.Memory.GPU)
	gpu.cmd_mem_copy(cmd_buf, verts_gpu, verts_staging)
	gpu.cmd_mem_copy(cmd_buf, indices_gpu, indices_staging)
	return verts_gpu, indices_gpu
}

draw_sky :: proc(
	cmd_buf: gpu.Command_Buffer,
	frame_arena: ^gpu.Arena,
	ctx: ^Gpu_Context,
	world_to_view: linalg.Matrix4f32,
	view_to_proj: linalg.Matrix4f32,
	output_srgb: bool,
) {
	data := gpu.arena_alloc(frame_arena, Sky_Draw_Data)
	data.cpu^ = {
		positions     = ctx.sky_vertices.gpu.ptr,
		world_to_view = intr.matrix_flatten(remove_view_translation(world_to_view)),
		view_to_proj  = intr.matrix_flatten(view_to_proj),
		sun_direction = {ctx.sun_direction.x, ctx.sun_direction.y, ctx.sun_direction.z, 0.0},
	}
	frag_data := gpu.arena_alloc(frame_arena, Sky_Frag_Data)
	frag_data.cpu^ = {
		sky_texture = ctx.sky_texture_id,
		sky_sampler = ctx.sky_sampler,
		output_srgb = 1 if output_srgb else 0,
	}
	gpu.cmd_set_shaders(cmd_buf, ctx.sky_vert, ctx.sky_frag)
	gpu.cmd_set_desc_heap(cmd_buf, ctx.desc_pool)
	gpu.cmd_set_depth_state(cmd_buf, {mode = {}, compare = .Always})
	gpu.cmd_set_raster_state(cmd_buf, {topology = .Triangle_List, cull_mode = .None})
	gpu.cmd_draw_indexed(cmd_buf, data, frag_data, ctx.sky_indices)
}

remove_view_translation :: proc(m: linalg.Matrix4f32) -> linalg.Matrix4f32 {
	res := m
	res[0, 3] = 0
	res[1, 3] = 0
	res[2, 3] = 0
	return res
}

draw_scene :: proc(
	cmd_buf: gpu.Command_Buffer,
	frame_arena: ^gpu.Arena,
	ctx: ^Gpu_Context,
	world_to_view: linalg.Matrix4f32,
	view_to_proj: linalg.Matrix4f32,
	camera_pos: [3]f32,
	output_srgb: bool,
) {
	gpu.cmd_set_shaders(cmd_buf, ctx.scene_vert, ctx.scene_frag)
	gpu.cmd_set_raster_state(cmd_buf, {topology = .Triangle_List, cull_mode = .None})
	gpu.cmd_set_desc_heap(cmd_buf, ctx.desc_pool)

	for instance in ctx.scene.instances {
		mesh := ctx.meshes[instance.mesh_idx]
		material := ctx.scene.meshes[instance.mesh_idx]
		data := gpu.arena_alloc(frame_arena, Scene_Draw_Data)
		data.cpu^ = {
			positions             = mesh.pos.gpu.ptr,
			normals               = mesh.normals.gpu.ptr,
			tangents              = mesh.tangents.gpu.ptr,
			uvs                   = mesh.uvs.gpu.ptr,
			model_to_world        = intr.matrix_flatten(instance.transform),
			model_to_world_normal = intr.matrix_flatten(
				linalg.transpose(linalg.inverse(instance.transform)),
			),
			world_to_view         = intr.matrix_flatten(world_to_view),
			view_to_proj          = intr.matrix_flatten(view_to_proj),
			light_origin          = vec3_to_vec4(ctx.light_projection.origin, 1.0),
			light_right           = vec3_to_vec4(ctx.light_projection.right, 0.0),
			light_up              = vec3_to_vec4(ctx.light_projection.up, 0.0),
			light_forward         = vec3_to_vec4(ctx.light_projection.forward, 0.0),
			light_extents         = {
				ctx.light_projection.width,
				ctx.light_projection.height,
				ctx.light_projection.near,
				ctx.light_projection.far,
			},
			camera_world_pos      = {camera_pos.x, camera_pos.y, camera_pos.z, 1.0},
		}
		frag_data := gpu.arena_alloc(frame_arena, Scene_Frag_Data)
		frag_data.cpu^ = {
			base_color_map                 = material.base_color_map,
			base_color_map_sampler         = ctx.sampler_id,
			metallic_roughness_map         = material.metallic_roughness_map,
			metallic_roughness_map_sampler = ctx.sampler_id,
			normal_map                     = material.normal_map,
			normal_map_sampler             = ctx.sampler_id,
			shadow_map                     = ctx.shadow_map_id,
			shadow_map_sampler             = ctx.shadow_sampler,
			sun_direction                  = {
				ctx.sun_direction.x,
				ctx.sun_direction.y,
				ctx.sun_direction.z,
				0.0,
			},
			light_extents                  = {
				ctx.light_projection.width,
				ctx.light_projection.height,
				ctx.light_projection.near,
				ctx.light_projection.far,
			},
			render_mode                    = 1 if Fullbright_Debug else 0,
			output_srgb                    = 1 if output_srgb else 0,
			sky_texture                    = ctx.sky_texture_id,
			sky_sampler                    = ctx.sky_sampler,
		}
		gpu.cmd_draw_indexed(cmd_buf, data, frag_data, mesh.indices)
	}
}

is_srgb_format :: proc(format: gpu.Texture_Format) -> bool {
	return format == .RGBA8_SRGB
}

render_shadow_pass :: proc(
	cmd_buf: gpu.Command_Buffer,
	frame_arena: ^gpu.Arena,
	ctx: ^Gpu_Context,
) {
	gpu.cmd_begin_render_pass(
		cmd_buf,
		{
			render_area_size = {Shadow_Size, Shadow_Size},
			depth_attachment = gpu.Render_Attachment{texture = ctx.shadow_map, clear_color = 1.0},
		},
	)
	gpu.cmd_set_shaders(cmd_buf, ctx.shadow_vert, ctx.shadow_frag)
	gpu.cmd_set_depth_state(cmd_buf, {mode = {.Read, .Write}, compare = .Less})
	gpu.cmd_set_raster_state(cmd_buf, {topology = .Triangle_List, cull_mode = .None})

	for instance in ctx.scene.instances {
		mesh := ctx.meshes[instance.mesh_idx]
		data := gpu.arena_alloc(frame_arena, Shadow_Draw_Data)
		data.cpu^ = {
			positions      = mesh.pos.gpu.ptr,
			model_to_world = intr.matrix_flatten(instance.transform),
			light_origin   = vec3_to_vec4(ctx.light_projection.origin, 1.0),
			light_right    = vec3_to_vec4(ctx.light_projection.right, 0.0),
			light_up       = vec3_to_vec4(ctx.light_projection.up, 0.0),
			light_forward  = vec3_to_vec4(ctx.light_projection.forward, 0.0),
			light_extents  = {
				ctx.light_projection.width,
				ctx.light_projection.height,
				ctx.light_projection.near,
				ctx.light_projection.far,
			},
		}
		gpu.cmd_draw_indexed(cmd_buf, data, {}, mesh.indices)
	}
	gpu.cmd_end_render_pass(cmd_buf)
	gpu.cmd_barrier(cmd_buf, .All, .Fragment_Shader, {})
}

update_sun :: proc(ctx: ^Gpu_Context, dt: f32) {
	ctx.sun_angle += dt * Sun_Spin_Rate
	ctx.sun_position =
		Sun_Target +
		[3]f32 {
				math.cos(ctx.sun_angle) * Sun_Orbit_Radius,
				Sun_Orbit_Height,
				math.sin(ctx.sun_angle) * Sun_Orbit_Radius,
			}
	ctx.sun_direction = linalg.normalize(Sun_Target - ctx.sun_position)
	ctx.light_projection = make_sun_projection(ctx.sun_position, ctx.sun_direction, ctx.scene_bounds_min, ctx.scene_bounds_max, ctx.scene_bounds_ok)
}

make_sun_projection :: proc(sun_pos, sun_dir, scene_min, scene_max: [3]f32, has_scene_bounds: bool) -> Light_Projection {
	forward := linalg.normalize(sun_dir)
	world_up := [3]f32{0, 1, 0}
	right := linalg.cross(forward, world_up)
	if linalg.length(right) < 0.001 {
		right = {1, 0, 0}
	} else {
		right = linalg.normalize(right)
	}
	up := linalg.normalize(linalg.cross(right, forward))
	if !has_scene_bounds {
		return {
			origin = sun_pos,
			right = right,
			up = up,
			forward = forward,
			width = 22.0,
			height = 14.0,
			near = 0.05,
			far = 42.0,
		}
	}

	corners := [?][3]f32 {
		{scene_min.x, scene_min.y, scene_min.z},
		{scene_max.x, scene_min.y, scene_min.z},
		{scene_min.x, scene_max.y, scene_min.z},
		{scene_max.x, scene_max.y, scene_min.z},
		{scene_min.x, scene_min.y, scene_max.z},
		{scene_max.x, scene_min.y, scene_max.z},
		{scene_min.x, scene_max.y, scene_max.z},
		{scene_max.x, scene_max.y, scene_max.z},
	}

	min_x, max_x := f32(max(f32)), -f32(max(f32))
	min_y, max_y := f32(max(f32)), -f32(max(f32))
	min_z, max_z := f32(max(f32)), -f32(max(f32))
	for corner in corners {
		x := linalg.dot(corner, right)
		y := linalg.dot(corner, up)
		z := linalg.dot(corner, forward)
		min_x = min(min_x, x)
		max_x = max(max_x, x)
		min_y = min(min_y, y)
		max_y = max(max_y, y)
		min_z = min(min_z, z)
		max_z = max(max_z, z)
	}

	margin_xy: f32 = 1.5
	margin_z: f32 = 6.0
	center_x := (min_x + max_x) * 0.5
	center_y := (min_y + max_y) * 0.5
	center_z := min_z - margin_z
	origin := right * center_x + up * center_y + forward * center_z
	return {
		origin = origin,
		right = right,
		up = up,
		forward = forward,
		width = max(1.0, max_x - min_x + margin_xy * 2.0),
		height = max(1.0, max_y - min_y + margin_xy * 2.0),
		near = 0.0,
		far = max(1.0, max_z - min_z + margin_z * 2.0),
	}
}

compute_scene_bounds :: proc(scene: shared.Scene) -> (bounds_min, bounds_max: [3]f32, ok: bool) {
	bounds_min = {f32(max(f32)), f32(max(f32)), f32(max(f32))}
	bounds_max = {-f32(max(f32)), -f32(max(f32)), -f32(max(f32))}

	for instance in scene.instances {
		mesh := scene.meshes[instance.mesh_idx]
		for pos in mesh.pos {
			world := transform_point(instance.transform, pos)
			bounds_min.x = min(bounds_min.x, world.x)
			bounds_min.y = min(bounds_min.y, world.y)
			bounds_min.z = min(bounds_min.z, world.z)
			bounds_max.x = max(bounds_max.x, world.x)
			bounds_max.y = max(bounds_max.y, world.y)
			bounds_max.z = max(bounds_max.z, world.z)
			ok = true
		}
	}

	return
}

transform_point :: proc(m: matrix[4, 4]f32, p: [4]f32) -> [3]f32 {
	return {
		m[0, 0] * p.x + m[0, 1] * p.y + m[0, 2] * p.z + m[0, 3] * p.w,
		m[1, 0] * p.x + m[1, 1] * p.y + m[1, 2] * p.z + m[1, 3] * p.w,
		m[2, 0] * p.x + m[2, 1] * p.y + m[2, 2] * p.z + m[2, 3] * p.w,
	}
}

vec3_to_vec4 :: proc(v: [3]f32, w: f32) -> [4]f32 {
	return {v.x, v.y, v.z, w}
}

draw_helper :: proc(
	cmd_buf: gpu.Command_Buffer,
	frame_arena: ^gpu.Arena,
	ctx: ^Gpu_Context,
	model_to_world: linalg.Matrix4f32,
	world_to_view: linalg.Matrix4f32,
	view_to_proj: linalg.Matrix4f32,
	color: [4]f32,
	first_index: int,
	index_count: int,
) {
	data := gpu.arena_alloc(frame_arena, Helper_Draw_Data)
	data.cpu^ = {
		positions      = ctx.helper_vertices.gpu.ptr,
		model_to_world = intr.matrix_flatten(model_to_world),
		world_to_view  = intr.matrix_flatten(world_to_view),
		view_to_proj   = intr.matrix_flatten(view_to_proj),
		color          = color,
	}
	gpu.cmd_set_shaders(cmd_buf, ctx.helper_vert, ctx.helper_frag)
	gpu.cmd_set_raster_state(cmd_buf, {topology = .Triangle_List, cull_mode = .None})
	gpu.cmd_draw_indexed_raw(
		cmd_buf,
		data,
		{},
		gpu.subslice(ctx.helper_indices, first_index).gpu,
		.U32,
		u32(index_count),
	)
}

render_helpers :: proc(
	cmd_buf: gpu.Command_Buffer,
	frame_arena: ^gpu.Arena,
	ctx: ^Gpu_Context,
	walker: ^Walker,
	xr_ctx: ^Xr_Context,
	world_to_view: linalg.Matrix4f32,
	view_to_proj: linalg.Matrix4f32,
) {
	if walker.teleport_active {
		start := teleport_start_world(walker, xr_ctx)
		end := walker.teleport_target
		dir := end - start
		dir.y = 0
		dist := linalg.length(dir)
		if dist > 0.001 {
			yaw := math.atan2(dir.x, dir.z)
			model :=
				linalg.matrix4_translate_f32(start) *
				linalg.matrix4_rotate_f32(yaw, {0, 1, 0}) *
				linalg.matrix4_scale_f32({1, 1, dist})
			draw_helper(
				cmd_buf,
				frame_arena,
				ctx,
				model,
				world_to_view,
				view_to_proj,
				{0.1, 0.85, 1.0, 0.85},
				0,
				6,
			)
		}
		draw_helper(
			cmd_buf,
			frame_arena,
			ctx,
			linalg.matrix4_translate_f32(walker.teleport_target),
			world_to_view,
			view_to_proj,
			{0.0, 1.0, 0.55, 0.9},
			6,
			6,
		)
	}

	for pose, i in xr_ctx.hand_poses {
		if !xr_ctx.hand_active[i] do continue
		p := pose_position(pose) + walker.xr_origin + xr_ctx.tracking_offset
		model :=
			linalg.matrix4_translate_f32(p) *
			linalg.matrix4_from_quaternion(pose_orientation(pose)) *
			linalg.matrix4_scale_f32({1.0, 1.0, 1.0})
		color := [4]f32{1.0, 0.72, 0.18, 1.0} if i == 0 else [4]f32{0.25, 0.68, 1.0, 1.0}
		draw_helper(cmd_buf, frame_arena, ctx, model, world_to_view, view_to_proj, color, 12, 36)
	}
}

xr_create_instance :: proc() -> Xr_Context {
	xr.load_base_procs()

	extensions := []cstring{xr.KHR_VULKAN_ENABLE_EXTENSION_NAME}
	app_info := xr.ApplicationInfo {
		applicationName    = xr.make_string(App_Name, xr.MAX_APPLICATION_NAME_SIZE),
		applicationVersion = 1,
		engineName         = xr.make_string("no_gfx", xr.MAX_ENGINE_NAME_SIZE),
		engineVersion      = 1,
		apiVersion         = xr.MAKE_VERSION(1, 0, 25),
	}
	instance_info := xr.InstanceCreateInfo {
		sType                 = .INSTANCE_CREATE_INFO,
		applicationInfo       = app_info,
		enabledExtensionCount = u32(len(extensions)),
		enabledExtensionNames = raw_data(extensions),
	}

	res: Xr_Context
	xr_check(xr.CreateInstance(&instance_info, &res.instance), "xrCreateInstance")
	xr.load_instance_procs(res.instance)
	return res
}

xr_try_get_system :: proc(ctx: ^Xr_Context) -> bool {
	system_info := xr.SystemGetInfo {
		sType      = .SYSTEM_GET_INFO,
		formFactor = .HEAD_MOUNTED_DISPLAY,
	}
	result := xr.GetSystem(ctx.instance, &system_info, &ctx.system_id)
	if result == .SUCCESS do return true
	if result == .ERROR_FORM_FACTOR_UNAVAILABLE do return false
	log.errorf("xrGetSystem failed: %v", result)
	ctx.setup_failed = true
	return false
}

xr_try_create_rendering :: proc(ctx: ^Xr_Context) -> bool {
	verify_result := verify_openxr_vulkan_device(ctx.instance, ctx.system_id)
	if verify_result != .SUCCESS {
		log.errorf("OpenXR Vulkan device verification failed: %v", verify_result)
		ctx.setup_failed = true
		return false
	}

	if !xr_create_session_space_and_swapchain(ctx) {
		ctx.setup_failed = true
		return false
	}
	if !xr_wrap_swapchain_images(ctx) {
		ctx.setup_failed = true
		return false
	}

	log.info("OpenXR rendering is ready; waiting for session READY.")
	return true
}

add_openxr_vulkan_device_extensions :: proc(instance: xr.Instance, system_id: xr.SystemId) {
	count: u32
	xr_check(
		xr.GetVulkanDeviceExtensionsKHR(instance, system_id, 0, &count, nil),
		"xrGetVulkanDeviceExtensionsKHR/count",
	)
	if count == 0 do return

	buffer := make([]u8, count)
	defer delete(buffer)
	xr_check(
		xr.GetVulkanDeviceExtensionsKHR(
			instance,
			system_id,
			count,
			&count,
			cstring(raw_data(buffer)),
		),
		"xrGetVulkanDeviceExtensionsKHR",
	)

	extension_string := string(buffer[:max(0, int(count) - 1)])
	extensions := strings.fields(extension_string)
	defer delete(extensions)
	for extension in extensions {
		gpu.vk_add_device_extension(strings.clone_to_cstring(extension))
	}
}

verify_openxr_vulkan_device :: proc(instance: xr.Instance, system_id: xr.SystemId) -> xr.Result {
	requirements := xr.GraphicsRequirementsVulkanKHR {
		sType = .GRAPHICS_REQUIREMENTS_VULKAN_KHR,
	}
	result := xr.GetVulkanGraphicsRequirementsKHR(instance, system_id, &requirements)
	if result != .SUCCESS do return result

	xr_physical_device: vk.PhysicalDevice
	result = xr.GetVulkanGraphicsDeviceKHR(
		instance,
		system_id,
		gpu.vk_get_instance(),
		&xr_physical_device,
	)
	if result != .SUCCESS do return result
	if xr_physical_device != gpu.vk_get_physical_device() {
		log.error("OpenXR runtime selected a different Vulkan physical device.")
		return .ERROR_RUNTIME_FAILURE
	}
	return .SUCCESS
}

xr_create_session_space_and_swapchain :: proc(ctx: ^Xr_Context) -> bool {
	binding := xr.GraphicsBindingVulkanKHR {
		sType            = .GRAPHICS_BINDING_VULKAN_KHR,
		instance         = gpu.vk_get_instance(),
		physicalDevice   = gpu.vk_get_physical_device(),
		device           = gpu.vk_get_device(),
		queueFamilyIndex = gpu.vk_get_queue_family(.Main),
		queueIndex       = 0,
	}
	session_info := xr.SessionCreateInfo {
		sType    = .SESSION_CREATE_INFO,
		next     = &binding,
		systemId = ctx.system_id,
	}
	result := xr.CreateSession(ctx.instance, &session_info, &ctx.session)
	if result != .SUCCESS {
		log.errorf("xrCreateSession failed: %v", result)
		return false
	}

	if !xr_create_app_space(ctx) do return false

	if !xr_setup_actions(ctx) do return false

	view_count: u32
	result = xr.EnumerateViewConfigurationViews(
		ctx.instance,
		ctx.system_id,
		.PRIMARY_STEREO,
		0,
		&view_count,
		nil,
	)
	if result != .SUCCESS {
		log.errorf("xrEnumerateViewConfigurationViews/count failed: %v", result)
		return false
	}
	ctx.config_views = make([]xr.ViewConfigurationView, view_count)
	for &view in ctx.config_views do view.sType = .VIEW_CONFIGURATION_VIEW
	result = xr.EnumerateViewConfigurationViews(
		ctx.instance,
		ctx.system_id,
		.PRIMARY_STEREO,
		view_count,
		&view_count,
		raw_data(ctx.config_views),
	)
	if result != .SUCCESS {
		log.errorf("xrEnumerateViewConfigurationViews failed: %v", result)
		return false
	}

	ctx.views = make([]xr.View, view_count)
	for &view in ctx.views do view.sType = .VIEW
	ctx.projection_views = make([]xr.CompositionLayerProjectionView, view_count)
	for &view in ctx.projection_views do view.sType = .COMPOSITION_LAYER_PROJECTION_VIEW

	vk_format, gpu_format, ok := choose_swapchain_format(ctx.session)
	if !ok do return false
	ctx.color_format = gpu_format

	swapchain_info := xr.SwapchainCreateInfo {
		sType       = .SWAPCHAIN_CREATE_INFO,
		usageFlags  = {.COLOR_ATTACHMENT, .TRANSFER_SRC},
		format      = i64(vk_format),
		sampleCount = ctx.config_views[0].recommendedSwapchainSampleCount,
		width       = ctx.config_views[0].recommendedImageRectWidth,
		height      = ctx.config_views[0].recommendedImageRectHeight,
		faceCount   = 1,
		arraySize   = view_count,
		mipCount    = 1,
	}
	result = xr.CreateSwapchain(ctx.session, &swapchain_info, &ctx.swapchain)
	if result != .SUCCESS {
		log.errorf("xrCreateSwapchain failed: %v", result)
		return false
	}

	ctx.depth_texture = gpu.texture_alloc_and_create(
		{
			type = .D2,
			dimensions = {
				ctx.config_views[0].recommendedImageRectWidth,
				ctx.config_views[0].recommendedImageRectHeight,
				1,
			},
			layer_count = view_count,
			sample_count = ctx.config_views[0].recommendedSwapchainSampleCount,
			format = .D32_Float,
			usage = {.Depth_Stencil_Attachment},
		},
	)
	return true
}

xr_create_app_space :: proc(ctx: ^Xr_Context) -> bool {
	space_info := xr.ReferenceSpaceCreateInfo {
		sType                = .REFERENCE_SPACE_CREATE_INFO,
		referenceSpaceType   = .STAGE,
		poseInReferenceSpace = identity_pose(),
	}
	result := xr.CreateReferenceSpace(ctx.session, &space_info, &ctx.app_space)
	if result == .SUCCESS {
		ctx.stage_space = true
		ctx.tracking_offset = {}
		log.info("Using OpenXR STAGE reference space for floor-relative tracking.")
		return true
	}

	log.warnf(
		"OpenXR STAGE reference space unavailable (%v); falling back to LOCAL with eye-height offset.",
		result,
	)
	space_info.referenceSpaceType = .LOCAL
	result = xr.CreateReferenceSpace(ctx.session, &space_info, &ctx.app_space)
	if result != .SUCCESS {
		log.errorf("xrCreateReferenceSpace(LOCAL) failed: %v", result)
		return false
	}
	ctx.stage_space = false
	ctx.tracking_offset = {0.0, Player_Eye_Height, 0.0}
	return true
}

xr_setup_actions :: proc(ctx: ^Xr_Context) -> bool {
	xr_string_to_path(ctx.instance, "/user/hand/left", &ctx.hand_paths[0])
	xr_string_to_path(ctx.instance, "/user/hand/right", &ctx.hand_paths[1])

	action_set_info := xr.ActionSetCreateInfo {
		sType                  = .ACTION_SET_CREATE_INFO,
		actionSetName          = xr.make_string("walker", xr.MAX_ACTION_SET_NAME_SIZE),
		localizedActionSetName = xr.make_string("Walker", xr.MAX_LOCALIZED_ACTION_SET_NAME_SIZE),
		priority               = 0,
	}
	result := xr.CreateActionSet(ctx.instance, &action_set_info, &ctx.action_set)
	if result != .SUCCESS {
		log.errorf("xrCreateActionSet failed: %v", result)
		return false
	}

	pose_info := xr.ActionCreateInfo {
		sType               = .ACTION_CREATE_INFO,
		actionName          = xr.make_string("aim_pose", xr.MAX_ACTION_NAME_SIZE),
		actionType          = .POSE_INPUT,
		countSubactionPaths = 2,
		subactionPaths      = raw_data(ctx.hand_paths[:]),
		localizedActionName = xr.make_string("Aim Pose", xr.MAX_LOCALIZED_ACTION_NAME_SIZE),
	}
	result = xr.CreateAction(ctx.action_set, &pose_info, &ctx.aim_pose_action)
	if result != .SUCCESS {
		log.errorf("xrCreateAction(aim_pose) failed: %v", result)
		return false
	}

	stick_info := xr.ActionCreateInfo {
		sType               = .ACTION_CREATE_INFO,
		actionName          = xr.make_string("left_stick", xr.MAX_ACTION_NAME_SIZE),
		actionType          = .VECTOR2F_INPUT,
		countSubactionPaths = 1,
		subactionPaths      = &ctx.hand_paths[0],
		localizedActionName = xr.make_string("Left Stick", xr.MAX_LOCALIZED_ACTION_NAME_SIZE),
	}
	result = xr.CreateAction(ctx.action_set, &stick_info, &ctx.left_stick)
	if result != .SUCCESS {
		log.errorf("xrCreateAction(left_stick) failed: %v", result)
		return false
	}

	xr_suggest_controller_bindings(
		ctx.instance,
		ctx.action_set,
		ctx.aim_pose_action,
		ctx.left_stick,
	)

	sets := [?]xr.ActionSet{ctx.action_set}
	attach_info := xr.SessionActionSetsAttachInfo {
		sType           = .SESSION_ACTION_SETS_ATTACH_INFO,
		countActionSets = 1,
		actionSets      = raw_data(sets[:]),
	}
	result = xr.AttachSessionActionSets(ctx.session, &attach_info)
	if result != .SUCCESS {
		log.errorf("xrAttachSessionActionSets failed: %v", result)
		return false
	}

	for path, i in ctx.hand_paths {
		space_info := xr.ActionSpaceCreateInfo {
			sType             = .ACTION_SPACE_CREATE_INFO,
			action            = ctx.aim_pose_action,
			subactionPath     = path,
			poseInActionSpace = identity_pose(),
		}
		result = xr.CreateActionSpace(ctx.session, &space_info, &ctx.aim_spaces[i])
		if result != .SUCCESS {
			log.warnf("xrCreateActionSpace(%d) failed: %v", i, result)
		}
	}
	return true
}

xr_suggest_controller_bindings :: proc(
	instance: xr.Instance,
	action_set: xr.ActionSet,
	aim_pose: xr.Action,
	left_stick: xr.Action,
) {
	profiles := [?]string {
		"/interaction_profiles/oculus/touch_controller",
		"/interaction_profiles/valve/index_controller",
		"/interaction_profiles/microsoft/motion_controller",
		"/interaction_profiles/htc/vive_controller",
		"/interaction_profiles/khr/simple_controller",
	}
	for profile in profiles {
		profile_path: xr.Path
		xr_string_to_path(instance, profile, &profile_path)

		left_aim, right_aim, thumbstick, trackpad: xr.Path
		xr_string_to_path(instance, "/user/hand/left/input/aim/pose", &left_aim)
		xr_string_to_path(instance, "/user/hand/right/input/aim/pose", &right_aim)
		xr_string_to_path(instance, "/user/hand/left/input/thumbstick", &thumbstick)
		xr_string_to_path(instance, "/user/hand/left/input/trackpad", &trackpad)

		bindings := [?]xr.ActionSuggestedBinding {
			{action = aim_pose, binding = left_aim},
			{action = aim_pose, binding = right_aim},
			{action = left_stick, binding = thumbstick},
			{action = left_stick, binding = trackpad},
		}
		info := xr.InteractionProfileSuggestedBinding {
			sType                  = .INTERACTION_PROFILE_SUGGESTED_BINDING,
			interactionProfile     = profile_path,
			countSuggestedBindings = u32(len(bindings)),
			suggestedBindings      = raw_data(bindings[:]),
		}
		result := xr.SuggestInteractionProfileBindings(instance, &info)
		if result != .SUCCESS {
			log.debugf("Skipping suggested bindings for %s: %v", profile, result)
		}
		_ = action_set
	}
}

xr_sync_input :: proc(ctx: ^Xr_Context, display_time: xr.Time) -> bool {
	if ctx.session_state != .FOCUSED {
		ctx.left_stick_value = {}
		for &active in ctx.hand_active do active = false
		return false
	}

	active_set := xr.ActiveActionSet {
		actionSet = ctx.action_set,
	}
	sync_info := xr.ActionsSyncInfo {
		sType                 = .ACTIONS_SYNC_INFO,
		countActiveActionSets = 1,
		activeActionSets      = &active_set,
	}
	result := xr.SyncActions(ctx.session, &sync_info)
	if result != .SUCCESS {
		if result != .SESSION_NOT_FOCUSED {
			log.warnf("xrSyncActions failed: %v", result)
		}
		ctx.left_stick_value = {}
		for &active in ctx.hand_active do active = false
		return false
	}

	stick_state := xr.ActionStateVector2f {
		sType = .ACTION_STATE_VECTOR2F,
	}
	xr.GetActionStateVector2f(
		ctx.session,
		&xr.ActionStateGetInfo {
			sType = .ACTION_STATE_GET_INFO,
			action = ctx.left_stick,
			subactionPath = ctx.hand_paths[0],
		},
		&stick_state,
	)
	ctx.left_stick_value =
		{stick_state.currentState.x, stick_state.currentState.y} if bool(stick_state.isActive) else {}

	for path, i in ctx.hand_paths {
		pose_state := xr.ActionStatePose {
			sType = .ACTION_STATE_POSE,
		}
		xr.GetActionStatePose(
			ctx.session,
			&xr.ActionStateGetInfo {
				sType = .ACTION_STATE_GET_INFO,
				action = ctx.aim_pose_action,
				subactionPath = path,
			},
			&pose_state,
		)
		ctx.hand_active[i] = false
		if !bool(pose_state.isActive) || ctx.aim_spaces[i] == {} do continue

		location := xr.SpaceLocation {
			sType = .SPACE_LOCATION,
		}
		loc_result := xr.LocateSpace(ctx.aim_spaces[i], ctx.app_space, display_time, &location)
		valid :=
			loc_result == .SUCCESS &&
			.ORIENTATION_VALID in location.locationFlags &&
			.POSITION_VALID in location.locationFlags
		if valid {
			ctx.hand_active[i] = true
			ctx.hand_poses[i] = location.pose
		}
	}
	return true
}

xr_wrap_swapchain_images :: proc(ctx: ^Xr_Context) -> bool {
	image_count: u32
	result := xr.EnumerateSwapchainImages(ctx.swapchain, 0, &image_count, nil)
	if result != .SUCCESS {
		log.errorf("xrEnumerateSwapchainImages/count failed: %v", result)
		return false
	}

	ctx.swapchain_images = make([]xr.SwapchainImageVulkanKHR, image_count)
	for &image in ctx.swapchain_images do image.sType = .SWAPCHAIN_IMAGE_VULKAN_KHR
	result = xr.EnumerateSwapchainImages(
		ctx.swapchain,
		image_count,
		&image_count,
		cast(^xr.SwapchainImageBaseHeader)raw_data(ctx.swapchain_images),
	)
	if result != .SUCCESS {
		log.errorf("xrEnumerateSwapchainImages failed: %v", result)
		return false
	}

	ctx.textures = make([]gpu.Texture, image_count)
	for image, i in ctx.swapchain_images {
		ctx.textures[i] = gpu.vk_wrap_image(
			image.image,
			{
				type = .D2,
				dimensions = {
					ctx.config_views[0].recommendedImageRectWidth,
					ctx.config_views[0].recommendedImageRectHeight,
					1,
				},
				layer_count = u32(len(ctx.config_views)),
				sample_count = ctx.config_views[0].recommendedSwapchainSampleCount,
				format = ctx.color_format,
				usage = {.Color_Attachment, .Transfer_Src},
			},
			name = fmt.tprintf("OpenXR swapchain image %d", i),
		)
	}
	return true
}

choose_swapchain_format :: proc(session: xr.Session) -> (vk.Format, gpu.Texture_Format, bool) {
	format_count: u32
	result := xr.EnumerateSwapchainFormats(session, 0, &format_count, nil)
	if result != .SUCCESS || format_count == 0 {
		log.errorf("xrEnumerateSwapchainFormats failed: %v", result)
		return {}, {}, false
	}
	formats := make([]i64, format_count)
	defer delete(formats)
	result = xr.EnumerateSwapchainFormats(session, format_count, &format_count, raw_data(formats))
	if result != .SUCCESS {
		log.errorf("xrEnumerateSwapchainFormats failed: %v", result)
		return {}, {}, false
	}

	candidates := []struct {
		vk:  vk.Format,
		gpu: gpu.Texture_Format,
	} {
		{vk = .R8G8B8A8_SRGB, gpu = .RGBA8_SRGB},
		{vk = .R8G8B8A8_UNORM, gpu = .RGBA8_Unorm},
		{vk = .B8G8R8A8_UNORM, gpu = .BGRA8_Unorm},
	}
	for candidate in candidates {
		for format in formats {
			if format == i64(candidate.vk) do return candidate.vk, candidate.gpu, true
		}
	}
	log.error("OpenXR runtime did not expose a supported color swapchain format.")
	return {}, {}, false
}

render_xr_frame :: proc(
	window: ^sdl.Window,
	gpu_ctx: ^Gpu_Context,
	ctx: ^Xr_Context,
	walker: ^Walker,
	frame_state: xr.FrameState,
) -> bool {
	locate_info := xr.ViewLocateInfo {
		sType                 = .VIEW_LOCATE_INFO,
		viewConfigurationType = .PRIMARY_STEREO,
		displayTime           = frame_state.predictedDisplayTime,
		space                 = ctx.app_space,
	}
	view_state := xr.ViewState {
		sType = .VIEW_STATE,
	}
	view_count: u32
	result := xr.LocateViews(
		ctx.session,
		&locate_info,
		&view_state,
		u32(len(ctx.views)),
		&view_count,
		raw_data(ctx.views),
	)
	if result != .SUCCESS {
		log.warnf("xrLocateViews failed: %v", result)
		xr_end_frame_empty(ctx.session, frame_state.predictedDisplayTime)
		return false
	}

	if xr_sync_input(ctx, frame_state.predictedDisplayTime) {
		update_xr_teleport(walker, ctx)
	} else {
		walker.teleport_active = false
		walker.teleport_stick_was_on = false
	}

	image_index: u32
	result = xr.AcquireSwapchainImage(
		ctx.swapchain,
		&xr.SwapchainImageAcquireInfo{sType = .SWAPCHAIN_IMAGE_ACQUIRE_INFO},
		&image_index,
	)
	if result != .SUCCESS {
		log.warnf("xrAcquireSwapchainImage failed: %v", result)
		xr_end_frame_empty(ctx.session, frame_state.predictedDisplayTime)
		return false
	}
	result = xr.WaitSwapchainImage(
		ctx.swapchain,
		&xr.SwapchainImageWaitInfo{sType = .SWAPCHAIN_IMAGE_WAIT_INFO, timeout = max(i64)},
	)
	if result != .SUCCESS {
		log.warnf("xrWaitSwapchainImage failed: %v", result)
		xr.ReleaseSwapchainImage(
			ctx.swapchain,
			&xr.SwapchainImageReleaseInfo{sType = .SWAPCHAIN_IMAGE_RELEASE_INFO},
		)
		xr_end_frame_empty(ctx.session, frame_state.predictedDisplayTime)
		return false
	}

	mirror_texture, mirror_ok := acquire_window_mirror(window, gpu_ctx)
	frame_arena := &gpu_ctx.frame_arenas[gpu_ctx.next_frame % Frames_In_Flight]
	gpu.arena_free_all(frame_arena)

	cmd_buf := gpu.commands_begin(.Main)
	render_shadow_pass(cmd_buf, frame_arena, gpu_ctx)
	for eye in 0 ..< int(view_count) {
		gpu.cmd_begin_render_pass(
			cmd_buf,
			{
				color_attachments = {
					{
						texture = ctx.textures[image_index],
						view = {base_layer = u16(eye), layer_count = 1},
						clear_color = {0.025, 0.028, 0.032, 1.0},
					},
				},
				depth_attachment = gpu.Render_Attachment {
					texture = ctx.depth_texture,
					view = {base_layer = u16(eye), layer_count = 1},
					clear_color = 1.0,
				},
			},
		)

		world_to_view := xr_world_to_view(
			ctx.views[eye].pose,
			walker.xr_origin + ctx.tracking_offset,
		)
		view_to_proj := fov_to_projection(ctx.views[eye].fov)
		camera_pos := walker.xr_origin + ctx.tracking_offset + pose_position(ctx.views[eye].pose)
		output_srgb := is_srgb_format(ctx.color_format)
		draw_sky(cmd_buf, frame_arena, gpu_ctx, world_to_view, view_to_proj, output_srgb)
		gpu.cmd_set_depth_state(cmd_buf, {mode = {.Read, .Write}, compare = .Less})
		draw_scene(cmd_buf, frame_arena, gpu_ctx, world_to_view, view_to_proj, camera_pos, output_srgb)
		render_helpers(cmd_buf, frame_arena, gpu_ctx, walker, ctx, world_to_view, view_to_proj)
		gpu.cmd_end_render_pass(cmd_buf)
	}

	if mirror_ok {
		gpu.cmd_barrier(cmd_buf, .All, .All, {})
		gpu.cmd_blit_texture(
			cmd_buf,
			mirror_texture,
			{},
			ctx.textures[image_index],
			{base_layer = 0, layer_count = 1},
			.Linear,
		)
	}

	gpu.cmd_add_signal_semaphore(cmd_buf, gpu_ctx.frame_sem, gpu_ctx.next_frame)
	gpu.queue_submit(.Main, {cmd_buf})
	gpu.semaphore_wait(gpu_ctx.frame_sem, gpu_ctx.next_frame)

	result = xr.ReleaseSwapchainImage(
		ctx.swapchain,
		&xr.SwapchainImageReleaseInfo{sType = .SWAPCHAIN_IMAGE_RELEASE_INFO},
	)
	if result != .SUCCESS {
		log.warnf("xrReleaseSwapchainImage failed: %v", result)
		xr_end_frame_empty(ctx.session, frame_state.predictedDisplayTime)
		return false
	}

	for i in 0 ..< int(view_count) {
		ctx.projection_views[i] = {
			sType = .COMPOSITION_LAYER_PROJECTION_VIEW,
			pose = ctx.views[i].pose,
			fov = ctx.views[i].fov,
			subImage = {
				swapchain = ctx.swapchain,
				imageRect = {
					offset = {},
					extent = {
						width = i32(ctx.config_views[0].recommendedImageRectWidth),
						height = i32(ctx.config_views[0].recommendedImageRectHeight),
					},
				},
				imageArrayIndex = u32(i),
			},
		}
	}

	projection_layer := xr.CompositionLayerProjection {
		sType     = .COMPOSITION_LAYER_PROJECTION,
		space     = ctx.app_space,
		viewCount = view_count,
		views     = raw_data(ctx.projection_views),
	}
	layer := cast(^xr.CompositionLayerBaseHeader)&projection_layer
	end_info := xr.FrameEndInfo {
		sType                = .FRAME_END_INFO,
		displayTime          = frame_state.predictedDisplayTime,
		environmentBlendMode = .OPAQUE,
		layerCount           = 1,
		layers               = &layer,
	}
	result = xr.EndFrame(ctx.session, &end_info)
	if result != .SUCCESS {
		log.warnf("xrEndFrame failed: %v", result)
		return false
	}

	if mirror_ok {
		gpu.swapchain_present(.Main, gpu_ctx.frame_sem, gpu_ctx.next_frame)
	}
	return true
}

update_xr_teleport :: proc(walker: ^Walker, ctx: ^Xr_Context) {
	stick := ctx.left_stick_value
	mag_sq := stick.x * stick.x + stick.y * stick.y
	stick_on := mag_sq > Teleport_Deadzone * Teleport_Deadzone
	if stick_on {
		aim_forward := teleport_forward(ctx)
		aim_forward.y = 0
		if linalg.length(aim_forward) < 0.001 do aim_forward = {0, 0, -1}
		move_dir := linalg.normalize(aim_forward)
		if linalg.length(move_dir) > 0.001 {
			pitch_amount := clamp(
				(abs(stick.y) - Teleport_Deadzone) / (1.0 - Teleport_Deadzone),
				0.0,
				1.0,
			)
			distance :=
				Teleport_Min_Range + (Teleport_Max_Range - Teleport_Min_Range) * pitch_amount
			start := teleport_start_world(walker, ctx)
			start.y = 0
			walker.teleport_target = start + move_dir * distance
			walker.teleport_target.y = 0
			head_floor := walker.xr_origin
			if len(ctx.views) > 0 {
				head_offset := ctx.tracking_offset + pose_position(ctx.views[0].pose)
				head_offset.y = 0
				head_floor = walker.xr_origin + head_offset
			}
			walker.teleport_player_target =
				walker.xr_origin + (walker.teleport_target - head_floor)
			walker.teleport_player_target.y = 0
			walker.teleport_active = true
		}
	} else if walker.teleport_stick_was_on && walker.teleport_active {
		walker.xr_origin = walker.teleport_player_target
		walker.teleport_active = false
	} else {
		walker.teleport_active = false
	}
	walker.teleport_stick_was_on = stick_on
}

teleport_start_world :: proc(walker: ^Walker, ctx: ^Xr_Context) -> [3]f32 {
	if ctx.hand_active[0] {
		return walker.xr_origin + ctx.tracking_offset + pose_position(ctx.hand_poses[0])
	}
	if len(ctx.views) > 0 {
		return walker.xr_origin + ctx.tracking_offset + pose_position(ctx.views[0].pose)
	}
	return walker.xr_origin + [3]f32{0, Player_Eye_Height, 0}
}

teleport_forward :: proc(ctx: ^Xr_Context) -> [3]f32 {
	if ctx.hand_active[0] {
		return linalg.quaternion_mul_vector3(pose_orientation(ctx.hand_poses[0]), [3]f32{0, 0, -1})
	}
	if len(ctx.views) > 0 {
		return linalg.quaternion_mul_vector3(pose_orientation(ctx.views[0].pose), [3]f32{0, 0, -1})
	}
	return {0, 0, -1}
}

acquire_window_mirror :: proc(window: ^sdl.Window, ctx: ^Gpu_Context) -> (gpu.Texture, bool) {
	old_size := ctx.window_size
	sdl.GetWindowSize(window, &ctx.window_size.x, &ctx.window_size.y)
	if .MINIMIZED in sdl.GetWindowFlags(window) ||
	   ctx.window_size.x <= 0 ||
	   ctx.window_size.y <= 0 {
		return {}, false
	}
	if old_size != ctx.window_size {
		gpu.queue_wait_idle(.Main)
		gpu.swapchain_resize({u32(max(1, ctx.window_size.x)), u32(max(1, ctx.window_size.y))})
		gpu_context_resize_window_depth(ctx)
	}
	return gpu.swapchain_acquire_next(), true
}

render_window_frame :: proc(window: ^sdl.Window, ctx: ^Gpu_Context, walker: ^Walker) {
	old_size := ctx.window_size
	sdl.GetWindowSize(window, &ctx.window_size.x, &ctx.window_size.y)
	if .MINIMIZED in sdl.GetWindowFlags(window) ||
	   ctx.window_size.x <= 0 ||
	   ctx.window_size.y <= 0 {
		sdl.Delay(16)
		return
	}

	if ctx.next_frame > Frames_In_Flight {
		gpu.semaphore_wait(ctx.frame_sem, ctx.next_frame - Frames_In_Flight)
	}
	if old_size != ctx.window_size {
		gpu.queue_wait_idle(.Main)
		gpu.swapchain_resize({u32(max(1, ctx.window_size.x)), u32(max(1, ctx.window_size.y))})
		gpu_context_resize_window_depth(ctx)
	}

	swapchain := gpu.swapchain_acquire_next()
	frame_arena := &ctx.frame_arenas[ctx.next_frame % Frames_In_Flight]
	gpu.arena_free_all(frame_arena)

	world_to_view := window_world_to_view(walker)
	aspect := f32(ctx.window_size.x) / f32(ctx.window_size.y)
	view_to_proj := linalg.matrix4_perspective_f32(
		math.RAD_PER_DEG * 65.0,
		aspect,
		0.05,
		1000.0,
		false,
	)

	cmd_buf := gpu.commands_begin(.Main)
	render_shadow_pass(cmd_buf, frame_arena, ctx)
	gpu.cmd_begin_render_pass(
		cmd_buf,
		{
			color_attachments = {{texture = swapchain, clear_color = {0.025, 0.028, 0.032, 1.0}}},
			depth_attachment = gpu.Render_Attachment {
				texture = ctx.window_depth,
				clear_color = 1.0,
			},
		},
	)
	draw_sky(cmd_buf, frame_arena, ctx, world_to_view, view_to_proj, false)
	gpu.cmd_set_depth_state(cmd_buf, {mode = {.Read, .Write}, compare = .Less})
	draw_scene(cmd_buf, frame_arena, ctx, world_to_view, view_to_proj, walker.window_pos, false)
	gpu.cmd_end_render_pass(cmd_buf)
	gpu.cmd_add_signal_semaphore(cmd_buf, ctx.frame_sem, ctx.next_frame)
	gpu.queue_submit(.Main, {cmd_buf})
	gpu.semaphore_wait(ctx.frame_sem, ctx.next_frame)
	gpu.swapchain_present(.Main, ctx.frame_sem, ctx.next_frame)
	ctx.next_frame += 1
}

update_window_walker :: proc(walker: ^Walker, input: ^Window_Input, dt: f32) {
	mouse_sensitivity: f32 = math.RAD_PER_DEG * 0.16
	if input.mouse_look {
		walker.window_yaw_pitch.x += input.mouse_dx * mouse_sensitivity
		walker.window_yaw_pitch.y += input.mouse_dy * mouse_sensitivity
		walker.window_yaw_pitch.y = clamp(
			walker.window_yaw_pitch.y,
			math.RAD_PER_DEG * -85.0,
			math.RAD_PER_DEG * 85.0,
		)
	}

	yaw := walker.window_yaw_pitch.x
	forward := [3]f32{math.sin(yaw), 0, -math.cos(yaw)}
	right := [3]f32{math.cos(yaw), 0, math.sin(yaw)}
	move: [3]f32
	move += forward * f32(int(input.keys[.W]) - int(input.keys[.S]))
	move += right * f32(int(input.keys[.A]) - int(input.keys[.D]))
	move.y += f32(int(input.keys[.E]) - int(input.keys[.Q]))
	if linalg.length(move) > 1 do move = linalg.normalize(move)
	speed: f32 = 5.0
	if input.keys[.LSHIFT] do speed = 12.0
	walker.window_pos += move * speed * dt
}

window_world_to_view :: proc(walker: ^Walker) -> linalg.Matrix4f32 {
	yaw := walker.window_yaw_pitch.x
	pitch := walker.window_yaw_pitch.y
	rot :=
		linalg.quaternion_angle_axis(yaw, [3]f32{0, 1, 0}) *
		linalg.quaternion_angle_axis(pitch, [3]f32{-1, 0, 0})
	view_rot := linalg.normalize(linalg.quaternion_inverse(rot))
	return(
		linalg.matrix4_from_quaternion(view_rot) *
		linalg.matrix4_translate_f32(-walker.window_pos) \
	)
}

xr_world_to_view :: proc(pose: xr.Posef, origin: [3]f32) -> linalg.Matrix4f32 {
	return pose_to_view_matrix(pose) * linalg.matrix4_translate_f32(-origin)
}

pose_to_view_matrix :: proc(pose: xr.Posef) -> linalg.Matrix4f32 {
	pose_to_world := linalg.matrix4_from_trs_f32(
		pose_position(pose),
		pose_orientation(pose),
		{1, 1, 1},
	)
	return linalg.inverse(pose_to_world)
}

pose_position :: proc(pose: xr.Posef) -> [3]f32 {
	return {pose.position.x, pose.position.y, pose.position.z}
}

pose_orientation :: proc(pose: xr.Posef) -> linalg.Quaternionf32 {
	return(
		transmute(linalg.Quaternionf32)[4]f32 {
			pose.orientation.x,
			pose.orientation.y,
			pose.orientation.z,
			pose.orientation.w,
		} \
	)
}

fov_to_projection :: proc(fov: xr.Fovf) -> linalg.Matrix4f32 {
	near: f32 = 0.05
	far: f32 = 1000.0
	tan_left := math.tan(fov.angleLeft)
	tan_right := math.tan(fov.angleRight)
	tan_up := math.tan(fov.angleUp)
	tan_down := math.tan(fov.angleDown)

	tan_width := tan_right - tan_left
	tan_height := tan_up - tan_down
	m := linalg.Matrix4f32{}
	m[0, 0] = 2.0 / tan_width
	m[0, 2] = (tan_right + tan_left) / tan_width
	m[1, 1] = 2.0 / tan_height
	m[1, 2] = (tan_up + tan_down) / tan_height
	m[2, 2] = -far / (far - near)
	m[2, 3] = -(far * near) / (far - near)
	m[3, 2] = -1.0
	return m
}

xr_end_frame_empty :: proc(session: xr.Session, display_time: xr.Time) {
	end_info := xr.FrameEndInfo {
		sType                = .FRAME_END_INFO,
		displayTime          = display_time,
		environmentBlendMode = .OPAQUE,
	}
	xr.EndFrame(session, &end_info)
}

poll_xr_events :: proc(ctx: ^Xr_Context) {
	for {
		event := xr.EventDataBuffer {
			sType = .EVENT_DATA_BUFFER,
		}
		result := xr.PollEvent(ctx.instance, &event)
		if result == .EVENT_UNAVAILABLE do break
		xr_check(result, "xrPollEvent")

		#partial switch event.sType {
		case .EVENT_DATA_SESSION_STATE_CHANGED:
			state_event := cast(^xr.EventDataSessionStateChanged)&event
			ctx.session_state = state_event.state
			log.infof("OpenXR session state: %v", state_event.state)
			#partial switch state_event.state {
			case .READY:
				begin_info := xr.SessionBeginInfo {
					sType                        = .SESSION_BEGIN_INFO,
					primaryViewConfigurationType = .PRIMARY_STEREO,
				}
				xr_check(xr.BeginSession(ctx.session, &begin_info), "xrBeginSession")
				ctx.session_running = true
			case .STOPPING:
				gpu.wait_idle()
				xr_check(xr.EndSession(ctx.session), "xrEndSession")
				ctx.session_running = false
			case .EXITING, .LOSS_PENDING:
				ctx.should_exit = true
			}
		}
	}
}

xr_destroy :: proc(ctx: ^Xr_Context) {
	if ctx.session != {} do gpu.wait_idle()
	for texture in ctx.textures {
		if texture.handle != nil do gpu.texture_destroy(texture)
	}
	for space in ctx.aim_spaces {
		if space != {} do xr.DestroySpace(space)
	}
	if ctx.depth_texture.handle != nil do gpu.texture_free_and_destroy(&ctx.depth_texture)
	delete(ctx.textures)
	delete(ctx.swapchain_images)
	delete(ctx.views)
	delete(ctx.config_views)
	delete(ctx.projection_views)
	if ctx.swapchain != {} do xr.DestroySwapchain(ctx.swapchain)
	if ctx.app_space != {} do xr.DestroySpace(ctx.app_space)
	if ctx.aim_pose_action != {} do xr.DestroyAction(ctx.aim_pose_action)
	if ctx.left_stick != {} do xr.DestroyAction(ctx.left_stick)
	if ctx.action_set != {} do xr.DestroyActionSet(ctx.action_set)
	if ctx.session != {} do xr.DestroySession(ctx.session)
	if ctx.instance != {} do xr.DestroyInstance(ctx.instance)
	ctx^ = {}
}

handle_sdl_events :: proc(window: ^sdl.Window, input: ^Window_Input) -> bool {
	input.mouse_dx = 0
	input.mouse_dy = 0
	event: sdl.Event
	for sdl.PollEvent(&event) {
		#partial switch event.type {
		case .QUIT:
			return false
		case .WINDOW_CLOSE_REQUESTED:
			if event.window.windowID == sdl.GetWindowID(window) do return false
		case .KEY_DOWN:
			if !event.key.repeat do input.keys[event.key.scancode] = true
		case .KEY_UP:
			if !event.key.repeat do input.keys[event.key.scancode] = false
		case .MOUSE_BUTTON_DOWN:
			if event.button.button == sdl.BUTTON_RIGHT do input.mouse_look = true
		case .MOUSE_BUTTON_UP:
			if event.button.button == sdl.BUTTON_RIGHT do input.mouse_look = false
		case .MOUSE_MOTION:
			input.mouse_dx += event.motion.xrel
			input.mouse_dy -= event.motion.yrel
		}
	}
	return true
}

identity_pose :: proc() -> xr.Posef {
	return {orientation = {x = 0, y = 0, z = 0, w = 1}, position = {}}
}

xr_string_to_path :: proc(instance: xr.Instance, path: string, out: ^xr.Path) {
	path_c := strings.clone_to_cstring(path)
	defer delete(path_c)
	xr_check(xr.StringToPath(instance, path_c, out), "xrStringToPath")
}

xr_check :: proc(result: xr.Result, what: string) {
	ensure(result == .SUCCESS, fmt.tprintf("%s failed: %v", what, result))
}
