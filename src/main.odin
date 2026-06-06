package main

import intr "base:intrinsics"
import "core:fmt"
import "core:log"
import "core:math"
import "core:math/linalg"
import "core:strings"

import gpu "../thirdparty/no_gfx/gpu"
import xr "../thirdparty/openxr"

import sdl "vendor:sdl3"
import vk "vendor:vulkan"

App_Name :: "OpenXR + no_gfx"
Frames_In_Flight :: 2

Gpu_Context :: struct {
	frame_sem:      gpu.Semaphore,
	next_frame:     u64,
	window_size:    [2]i32,
	shader_vert:    gpu.Shader,
	shader_frag:    gpu.Shader,
	cube_positions: gpu.slice_t([4]f32),
	cube_normals:   gpu.slice_t([4]f32),
	cube_indices:   gpu.slice_t(u32),
	window_depth:   gpu.Owned_Texture,
	frame_arenas:   [Frames_In_Flight]gpu.Arena,
	cube_time:      f32,
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
	setup_failed:     bool,
	should_exit:      bool,
}

main :: proc() {
	ok := sdl.Init({.VIDEO})
	ensure(ok, fmt.tprintf("SDL_Init(SDL_INIT_VIDEO) failed: %v", sdl.GetError()))
	defer sdl.Quit()

	console_logger := log.create_console_logger()
	defer log.destroy_console_logger(console_logger)
	context.logger = console_logger

	window := sdl.CreateWindow(App_Name, 960, 540, {.VULKAN, .RESIZABLE})
	ensure(window != nil, fmt.tprintf("SDL_CreateWindow failed: %v", sdl.GetError()))
	defer sdl.DestroyWindow(window)

	xr_ctx := xr_create_instance()

	if xr_try_get_system(&xr_ctx) {
		add_openxr_vulkan_device_extensions(xr_ctx.instance, xr_ctx.system_id)
	} else {
		log.info("No OpenXR HMD system is available yet")
	}

	gpu_ctx := gpu_init_for_window(window)
	defer gpu_context_destroy(&gpu_ctx)
	defer xr_destroy(&xr_ctx)

	ts_freq := sdl.GetPerformanceFrequency()
	now_ts := sdl.GetPerformanceCounter()
	retry_frame: u64
	for !xr_ctx.should_exit {
		last_ts := now_ts
		now_ts = sdl.GetPerformanceCounter()
		delta_time := min(0.1, f32(f64(now_ts - last_ts) / f64(ts_freq)))
		gpu_ctx.cube_time += delta_time

		if !handle_sdl_events(window) {
			xr_ctx.should_exit = true
		}

		if xr_ctx.instance != {} {
			poll_xr_events(&xr_ctx)
		}

		if !xr_ctx.render_ready && !xr_ctx.setup_failed && retry_frame == 0 {
			if xr_ctx.system_id == {} && xr_try_get_system(&xr_ctx) {
				log.info("OpenXR HMD system became available; recreating device.")
				gpu_context_destroy(&gpu_ctx)
				add_openxr_vulkan_device_extensions(xr_ctx.instance, xr_ctx.system_id)
				gpu_ctx = gpu_init_for_window(window)
			}

			if xr_ctx.system_id != {} {
				xr_ctx.render_ready = xr_try_create_rendering(&xr_ctx)
			}
		}
		retry_frame = (retry_frame + 1) % 60

		if xr_ctx.session_running {
			frame_state := xr.FrameState {
				sType = .FRAME_STATE,
			}
			xr_check(
				xr.WaitFrame(
					xr_ctx.session,
					&xr.FrameWaitInfo{sType = .FRAME_WAIT_INFO},
					&frame_state,
				),
				"xrWaitFrame",
			)
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

				render_xr_frame(window, &gpu_ctx, &xr_ctx, frame_state)
				gpu_ctx.next_frame += 1
			}
		} else {
			render_window_frame(window, &gpu_ctx)
		}
	}

	gpu.wait_idle()
}

gpu_init_for_window :: proc(window: ^sdl.Window) -> Gpu_Context {
	ensure(gpu.init(), "gpu.init() failed!!")
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
		for &arena in ctx.frame_arenas do gpu.arena_destroy(&arena)
		if ctx.cube_positions.gpu.ptr != nil do gpu.mem_free(ctx.cube_positions)
		if ctx.cube_normals.gpu.ptr != nil do gpu.mem_free(ctx.cube_normals)
		if ctx.cube_indices.gpu.ptr != nil do gpu.mem_free(ctx.cube_indices)
		if ctx.shader_vert != nil do gpu.shader_destroy(ctx.shader_vert)
		if ctx.shader_frag != nil do gpu.shader_destroy(ctx.shader_frag)
		gpu.semaphore_destroy(ctx.frame_sem)
		gpu.cleanup()
	}
	ctx^ = {}
}

gpu_context_create_scene_resources :: proc(ctx: ^Gpu_Context) {
	ctx.shader_vert = gpu.shader_create(#load("shaders/cube.vert.spv", []u32), .Vertex)
	ctx.shader_frag = gpu.shader_create(#load("shaders/cube.frag.spv", []u32), .Fragment)

	upload_arena := gpu.arena_init()
	defer gpu.arena_destroy(&upload_arena)

	positions, normals, indices := cube_mesh_data()

	positions_staging := gpu.arena_alloc(&upload_arena, [4]f32, len(positions))
	normals_staging := gpu.arena_alloc(&upload_arena, [4]f32, len(normals))
	indices_staging := gpu.arena_alloc(&upload_arena, u32, len(indices))
	copy(positions_staging.cpu, positions[:])
	copy(normals_staging.cpu, normals[:])
	copy(indices_staging.cpu, indices[:])

	ctx.cube_positions = gpu.mem_alloc([4]f32, len(positions), gpu.Memory.GPU)
	ctx.cube_normals = gpu.mem_alloc([4]f32, len(normals), gpu.Memory.GPU)
	ctx.cube_indices = gpu.mem_alloc(u32, len(indices), gpu.Memory.GPU)

	upload_cmd_buf := gpu.commands_begin(.Main)
	gpu.cmd_mem_copy(upload_cmd_buf, ctx.cube_positions, positions_staging)
	gpu.cmd_mem_copy(upload_cmd_buf, ctx.cube_normals, normals_staging)
	gpu.cmd_mem_copy(upload_cmd_buf, ctx.cube_indices, indices_staging)
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
		extension_c := strings.clone_to_cstring(extension)
		gpu.vk_add_device_extension(extension_c)
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

	space_info := xr.ReferenceSpaceCreateInfo {
		sType                = .REFERENCE_SPACE_CREATE_INFO,
		referenceSpaceType   = .LOCAL,
		poseInReferenceSpace = identity_pose(),
	}
	result = xr.CreateReferenceSpace(ctx.session, &space_info, &ctx.app_space)
	if result != .SUCCESS {
		log.errorf("xrCreateReferenceSpace failed: %v", result)
		return false
	}

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
	if result != .SUCCESS {
		log.errorf("xrEnumerateSwapchainFormats/count failed: %v", result)
		return {}, {}, false
	}
	if format_count == 0 {
		log.error("OpenXR runtime reported no swapchain formats.")
		return {}, {}, false
	}

	formats := make([]i64, format_count)
	result = xr.EnumerateSwapchainFormats(session, format_count, &format_count, raw_data(formats))
	if result != .SUCCESS {
		log.errorf("xrEnumerateSwapchainFormats failed: %v", result)
		delete(formats)
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
			if format == i64(candidate.vk) {
				delete(formats)
				return candidate.vk, candidate.gpu, true
			}
		}
	}

	delete(formats)
	log.error(
		"OpenXR runtime did not expose a color swapchain format supported by this application.",
	)
	return {}, {}, false
}

render_xr_frame :: proc(
	window: ^sdl.Window,
	gpu_ctx: ^Gpu_Context,
	ctx: ^Xr_Context,
	frame_state: xr.FrameState,
) {
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
	xr_check(
		xr.LocateViews(
			ctx.session,
			&locate_info,
			&view_state,
			u32(len(ctx.views)),
			&view_count,
			raw_data(ctx.views),
		),
		"xrLocateViews",
	)

	image_index: u32
	xr_check(
		xr.AcquireSwapchainImage(
			ctx.swapchain,
			&xr.SwapchainImageAcquireInfo{sType = .SWAPCHAIN_IMAGE_ACQUIRE_INFO},
			&image_index,
		),
		"xrAcquireSwapchainImage",
	)
	xr_check(
		xr.WaitSwapchainImage(
			ctx.swapchain,
			&xr.SwapchainImageWaitInfo{sType = .SWAPCHAIN_IMAGE_WAIT_INFO, timeout = max(i64)},
		),
		"xrWaitSwapchainImage",
	)

	mirror_texture, mirror_ok := acquire_window_mirror(window, gpu_ctx)

	frame_arena := &gpu_ctx.frame_arenas[gpu_ctx.next_frame % Frames_In_Flight]
	gpu.arena_free_all(frame_arena)

	cmd_buf := gpu.commands_begin(.Main)
	for eye in 0 ..< int(view_count) {
		gpu.cmd_begin_render_pass(
			cmd_buf,
			{
				color_attachments = {
					{
						texture = ctx.textures[image_index],
						view = {base_layer = u16(eye), layer_count = 1},
						clear_color = {0.015, 0.018, 0.024, 1.0},
					},
				},
				depth_attachment = gpu.Render_Attachment {
					texture = ctx.depth_texture,
					view = {base_layer = u16(eye), layer_count = 1},
					clear_color = 1.0,
				},
			},
		)
		gpu.cmd_set_depth_state(cmd_buf, {mode = {.Read, .Write}, compare = .Less})
		draw_cube(
			cmd_buf,
			frame_arena,
			gpu_ctx,
			pose_to_view_matrix(ctx.views[eye].pose),
			fov_to_projection(ctx.views[eye].fov),
			pose_position(ctx.views[eye].pose),
		)
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

	xr_check(
		xr.ReleaseSwapchainImage(
			ctx.swapchain,
			&xr.SwapchainImageReleaseInfo{sType = .SWAPCHAIN_IMAGE_RELEASE_INFO},
		),
		"xrReleaseSwapchainImage",
	)

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
	xr_check(xr.EndFrame(ctx.session, &end_info), "xrEndFrame")

	if mirror_ok {
		gpu.swapchain_present(.Main, gpu_ctx.frame_sem, gpu_ctx.next_frame)
	}
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
		gpu.swapchain_resize({u32(max(0, ctx.window_size.x)), u32(max(0, ctx.window_size.y))})
		gpu_context_resize_window_depth(ctx)
	}

	return gpu.swapchain_acquire_next(), true
}

render_window_frame :: proc(window: ^sdl.Window, ctx: ^Gpu_Context) {
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
		gpu.swapchain_resize({u32(max(0, ctx.window_size.x)), u32(max(0, ctx.window_size.y))})
		gpu_context_resize_window_depth(ctx)
	}

	swapchain := gpu.swapchain_acquire_next()
	frame_arena := &ctx.frame_arenas[ctx.next_frame % Frames_In_Flight]
	gpu.arena_free_all(frame_arena)

	camera_pos := [3]f32{0.0, 0.0, 3.2}
	world_to_view := linalg.matrix4_look_at_f32(
		camera_pos,
		{0.0, 0.0, -3.0},
		{0.0, 1.0, 0.0},
		false,
	)
	aspect := f32(ctx.window_size.x) / f32(ctx.window_size.y)
	view_to_proj := linalg.matrix4_perspective_f32(
		math.RAD_PER_DEG * 65.0,
		aspect,
		0.05,
		100.0,
		false,
	)

	cmd_buf := gpu.commands_begin(.Main)
	gpu.cmd_begin_render_pass(
		cmd_buf,
		{
			color_attachments = {{texture = swapchain, clear_color = {0.015, 0.018, 0.024, 1.0}}},
			depth_attachment = gpu.Render_Attachment {
				texture = ctx.window_depth,
				clear_color = 1.0,
			},
		},
	)
	gpu.cmd_set_depth_state(cmd_buf, {mode = {.Read, .Write}, compare = .Less})
	draw_cube(cmd_buf, frame_arena, ctx, world_to_view, view_to_proj, camera_pos)
	gpu.cmd_end_render_pass(cmd_buf)
	gpu.cmd_add_signal_semaphore(cmd_buf, ctx.frame_sem, ctx.next_frame)
	gpu.queue_submit(.Main, {cmd_buf})
	gpu.swapchain_present(.Main, ctx.frame_sem, ctx.next_frame)
	ctx.next_frame += 1
}

Cube_Draw_Data :: struct #all_or_none {
	positions:             rawptr,
	normals:               rawptr,
	model_to_world:        [16]f32,
	model_to_world_normal: [16]f32,
	world_to_view:         [16]f32,
	view_to_proj:          [16]f32,
	camera_world_pos:      [4]f32,
}

draw_cube :: proc(
	cmd_buf: gpu.Command_Buffer,
	frame_arena: ^gpu.Arena,
	ctx: ^Gpu_Context,
	world_to_view: linalg.Matrix4f32,
	view_to_proj: linalg.Matrix4f32,
	camera_pos: [3]f32,
) {
    translation := linalg.matrix4_translate_f32({0.0, 0.0, -3.0})
    rotation := linalg.mul(
        linalg.matrix4_rotate_f32(ctx.cube_time * 0.85, {0.0, 1.0, 0.0}),
        linalg.matrix4_rotate_f32(ctx.cube_time * 0.47, {1.0, 0.0, 0.0}),
    )
    model_to_world := linalg.mul(translation, rotation)
    model_to_world_normal := linalg.transpose(linalg.inverse(model_to_world))

	data := gpu.arena_alloc(frame_arena, Cube_Draw_Data)
	data.cpu^ = {
		positions             = ctx.cube_positions.gpu.ptr,
		normals               = ctx.cube_normals.gpu.ptr,
		model_to_world        = intr.matrix_flatten(model_to_world),
		model_to_world_normal = intr.matrix_flatten(model_to_world_normal),
		world_to_view         = intr.matrix_flatten(world_to_view),
		view_to_proj          = intr.matrix_flatten(view_to_proj),
		camera_world_pos      = {camera_pos.x, camera_pos.y, camera_pos.z, 1.0},
	}

	gpu.cmd_set_shaders(cmd_buf, ctx.shader_vert, ctx.shader_frag)
	gpu.cmd_set_raster_state(cmd_buf, {cull_mode = .Cull_CCW})
	gpu.cmd_draw_indexed(cmd_buf, data, {}, ctx.cube_indices)
}

pose_to_view_matrix :: proc(pose: xr.Posef) -> linalg.Matrix4f32 {
	pose_to_world := linalg.matrix4_from_trs_f32(
		pose_position(pose),
		pose_orientation(pose),
		{1.0, 1.0, 1.0},
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
    // asymmetric projection matrix
    near: f32 = 0.05
    far:  f32 = 100.0

    tan_left  := math.tan(fov.angleLeft)
    tan_right := math.tan(fov.angleRight)
    tan_up    := math.tan(fov.angleUp)
    tan_down  := math.tan(fov.angleDown)

    tan_width  := tan_right - tan_left
    tan_height := tan_down - tan_up

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
	xr_check(xr.EndFrame(session, &end_info), "xrEndFrame/empty")
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

			#partial switch state_event.state {
			case .READY:
				begin_info := xr.SessionBeginInfo {
					sType                        = .SESSION_BEGIN_INFO,
					primaryViewConfigurationType = .PRIMARY_STEREO,
				}
				xr_check(xr.BeginSession(ctx.session, &begin_info), "xrBeginSession")
				ctx.session_running = true
			case .STOPPING:
				xr_check(xr.EndSession(ctx.session), "xrEndSession")
				ctx.session_running = false
			case .EXITING, .LOSS_PENDING:
				ctx.should_exit = true
			}
		}
	}
}

xr_destroy :: proc(ctx: ^Xr_Context) {
	if ctx.session != {} {
		gpu.wait_idle()
	}
	for texture in ctx.textures {
		if texture.handle != nil {
			gpu.texture_destroy(texture)
		}
	}
	if ctx.depth_texture.handle != nil {
		gpu.texture_free_and_destroy(&ctx.depth_texture)
	}
	delete(ctx.textures)
	delete(ctx.swapchain_images)
	delete(ctx.views)
	delete(ctx.config_views)
	delete(ctx.projection_views)
	if ctx.swapchain != {} do xr.DestroySwapchain(ctx.swapchain)
	if ctx.app_space != {} do xr.DestroySpace(ctx.app_space)
	if ctx.session != {} do xr.DestroySession(ctx.session)
	if ctx.instance != {} do xr.DestroyInstance(ctx.instance)
	ctx^ = {}
}

handle_sdl_events :: proc(window: ^sdl.Window) -> bool {
	event: sdl.Event
	for sdl.PollEvent(&event) {
		#partial switch event.type {
		case .QUIT:
			return false
		case .WINDOW_CLOSE_REQUESTED:
			if event.window.windowID == sdl.GetWindowID(window) {
				return false
			}
		}
	}
	return true
}

identity_pose :: proc() -> xr.Posef {
	return {orientation = {x = 0, y = 0, z = 0, w = 1}, position = {}}
}

xr_check :: proc(result: xr.Result, what: string) {
	ensure(result == .SUCCESS, fmt.tprintf("%s failed: %v", what, result))
}

cube_mesh_data :: proc() -> (positions: [24][4]f32, normals: [24][4]f32, indices: [36]u32) {
    // odinfmt: disable
    positions = {
        {-1, -1,  1, 1}, { 1, -1,  1, 1}, { 1,  1,  1, 1}, {-1,  1,  1, 1},
        { 1, -1, -1, 1}, {-1, -1, -1, 1}, {-1,  1, -1, 1}, { 1,  1, -1, 1},
        {-1, -1, -1, 1}, {-1, -1,  1, 1}, {-1,  1,  1, 1}, {-1,  1, -1, 1},
        { 1, -1,  1, 1}, { 1, -1, -1, 1}, { 1,  1, -1, 1}, { 1,  1,  1, 1},
        {-1,  1,  1, 1}, { 1,  1,  1, 1}, { 1,  1, -1, 1}, {-1,  1, -1, 1},
        {-1, -1, -1, 1}, { 1, -1, -1, 1}, { 1, -1,  1, 1}, {-1, -1,  1, 1},
    }
    normals = {
        { 0,  0,  1, 0}, { 0,  0,  1, 0}, { 0,  0,  1, 0}, { 0,  0,  1, 0},
        { 0,  0, -1, 0}, { 0,  0, -1, 0}, { 0,  0, -1, 0}, { 0,  0, -1, 0},
        {-1,  0,  0, 0}, {-1,  0,  0, 0}, {-1,  0,  0, 0}, {-1,  0,  0, 0},
        { 1,  0,  0, 0}, { 1,  0,  0, 0}, { 1,  0,  0, 0}, { 1,  0,  0, 0},
        { 0,  1,  0, 0}, { 0,  1,  0, 0}, { 0,  1,  0, 0}, { 0,  1,  0, 0},
        { 0, -1,  0, 0}, { 0, -1,  0, 0}, { 0, -1,  0, 0}, { 0, -1,  0, 0},
    }
    indices = {
         0,  1,  2,  0,  2,  3,
         4,  5,  6,  4,  6,  7,
         8,  9, 10,  8, 10, 11,
        12, 13, 14, 12, 14, 15,
        16, 17, 18, 16, 18, 19,
        20, 21, 22, 20, 22, 23,
    }
    // odinfmt: enable
	return
}
