from OpenGL.GL import *
from glfw.GLFW import *
import glm
import ctypes
import numpy as np
import os
import sys

# --------------------------------------------------------
# 1. 카메라 및 전역 변수
# --------------------------------------------------------
# 태양계 전체를 잘 조망하기 위해 카메라 거리를 더 멀리 띄웠습니다.
g_cam_r = 35.0  
g_cam_theta = np.radians(45)
g_cam_phi = np.radians(45)

g_P = glm.mat4()
g_height = 1000
g_width = 1000

g_cam_center = glm.vec3(0.0,0.0,0.0)
g_mouse_is_dragged = False
g_mouse_x_pos = 0.0
g_mouse_y_pos = 0.0
g_z_is_pressed = False
g_x_is_pressed = False


# --------------------------------------------------------
# 2. 셰이더 소스 코드
# --------------------------------------------------------

# --- (1) 단순 프레임, 그리드를 그리기 위한 Vertex Color 셰이더 ---
g_vertex_shader_src_color_attribute = '''
#version 330 core
layout (location = 0) in vec3 vin_pos; 
layout (location = 1) in vec3 vin_color; 
out vec4 vout_color;
uniform mat4 MVP;
void main()
{
    vec4 p3D_in_hcoord = vec4(vin_pos.xyz, 1.0);
    gl_Position = MVP * p3D_in_hcoord;
    vout_color = vec4(vin_color, 1.);
}
'''

# --- (2) 광원(태양) 등 조명 없이 단일 색상을 칠하기 위한 셰이더 ---
g_vertex_shader_src_color_uniform = '''
#version 330 core
layout (location = 0) in vec3 vin_pos; 
out vec4 vout_color;
uniform mat4 MVP;
uniform vec3 color;
void main()
{
    vec4 p3D_in_hcoord = vec4(vin_pos.xyz, 1.0);
    gl_Position = MVP * p3D_in_hcoord;
    vout_color = vec4(color, 1.);
}
'''

g_fragment_shader_src_color = '''
#version 330 core
in vec4 vout_color;
out vec4 FragColor;
void main()
{
    FragColor = vout_color;
}
'''

# --- (3) 조명을 받는 물체(행성, 달, 파이프)를 위한 Phong 셰이더 ---
g_vertex_shader_src_phong = '''
#version 330 core
layout (location = 0) in vec3 vin_pos; 
layout (location = 1) in vec3 vin_normal; 

out vec3 vout_surface_pos;
out vec3 vout_normal;

uniform mat4 MVP;
uniform mat4 M;

void main()
{
    vec4 p3D_in_hcoord = vec4(vin_pos.xyz, 1.0);
    gl_Position = MVP * p3D_in_hcoord;

    vout_surface_pos = vec3(M * vec4(vin_pos, 1));
    vout_normal = normalize( mat3(inverse(transpose(M)) ) * vin_normal);
}
'''

g_fragment_shader_src_phong = '''
#version 330 core
in vec3 vout_surface_pos;
in vec3 vout_normal;

out vec4 FragColor;

uniform vec3 view_pos;
uniform vec3 light_pos;
uniform vec3 object_color;

void main()
{
    vec3 light_color = vec3(1,1,1);
    float material_shininess = 32.0;

    vec3 light_ambient = 0.1*light_color;
    vec3 light_diffuse = light_color;
    vec3 light_specular = light_color;

    vec3 material_ambient = object_color;
    vec3 material_diffuse = object_color;
    vec3 material_specular = vec3(1,1,1);

    vec3 ambient = light_ambient * material_ambient;

    vec3 normal = normalize(vout_normal);
    vec3 surface_pos = vout_surface_pos;
    vec3 light_dir = normalize(light_pos - surface_pos);

    float diff = max(dot(normal, light_dir), 0);
    vec3 diffuse = diff * light_diffuse * material_diffuse;

    vec3 view_dir = normalize(view_pos - surface_pos);
    vec3 reflect_dir = reflect(-light_dir, normal);
    float spec = pow( max(dot(view_dir, reflect_dir), 0.0), material_shininess);
    vec3 specular = spec * light_specular * material_specular;

    vec3 color = ambient + diffuse + specular;
    FragColor = vec4(color, 1.);
}
'''


# --------------------------------------------------------
# 3. 계층적 모델링 Node 클래스
# --------------------------------------------------------
class Node:
    def __init__(self, parent, shape_transform, color, vao, vertex_count):
        self.parent = parent
        self.children = []
        if parent is not None:
            parent.children.append(self)

        self.transform = glm.mat4()
        self.global_transform = glm.mat4()

        self.shape_transform = shape_transform
        self.color = color
        self.vao = vao
        self.vertex_count = vertex_count

    def set_transform(self, transform):
        self.transform = transform

    def update_tree_global_transform(self):
        if self.parent is not None:
            self.global_transform = self.parent.get_global_transform() * self.transform
        else:
            self.global_transform = self.transform

        for child in self.children:
            child.update_tree_global_transform()

    def get_global_transform(self): return self.global_transform
    def get_shape_transform(self): return self.shape_transform
    def get_color(self): return self.color
    def get_vao(self): return self.vao
    def get_vertex_count(self): return self.vertex_count


# --------------------------------------------------------
# 4. 함수들 (Shaders, Callbacks, VAO 준비 함수)
# --------------------------------------------------------
def load_shaders(vertex_shader_source, fragment_shader_source):
    vertex_shader = glCreateShader(GL_VERTEX_SHADER)
    glShaderSource(vertex_shader, vertex_shader_source)
    glCompileShader(vertex_shader)
    
    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
    glShaderSource(fragment_shader, fragment_shader_source)
    glCompileShader(fragment_shader)
    
    shader_program = glCreateProgram()
    glAttachShader(shader_program, vertex_shader)
    glAttachShader(shader_program, fragment_shader)
    glLinkProgram(shader_program)
        
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)

    return shader_program

def button_callback(window, button, action, mod):
    global g_mouse_is_dragged, g_mouse_x_pos, g_mouse_y_pos
    if button==GLFW_MOUSE_BUTTON_LEFT:
        if action==GLFW_PRESS:
            g_mouse_is_dragged = True
            g_mouse_x_pos, g_mouse_y_pos = glfwGetCursorPos(window)
        elif action==GLFW_RELEASE:
            g_mouse_is_dragged = False

def key_callback(window, key, scancode, action, mods):
    global g_x_is_pressed, g_z_is_pressed
    if key==GLFW_KEY_ESCAPE and action==GLFW_PRESS:
        glfwSetWindowShouldClose(window, GLFW_TRUE)
    if key == GLFW_KEY_X:
        if action == GLFW_PRESS or action == GLFW_REPEAT:
            g_x_is_pressed = True
        elif action == GLFW_RELEASE:
            g_x_is_pressed = False
    elif key == GLFW_KEY_Z:
        if action == GLFW_PRESS or action == GLFW_REPEAT:
            g_z_is_pressed = True
        elif action == GLFW_RELEASE:
            g_z_is_pressed = False

def cursor_callback(window, xpos, ypos):
    global g_cam_r, g_cam_theta, g_cam_phi, g_cam_center, g_mouse_is_dragged, g_mouse_x_pos, g_mouse_y_pos,g_x_is_pressed, g_z_is_pressed
    if g_mouse_is_dragged:
        dx = xpos - g_mouse_x_pos
        dy = ypos - g_mouse_y_pos
        if g_x_is_pressed:
            pan_sens = 0.01
            front_dir = glm.vec3(np.sin(g_cam_theta), 0.0, np.cos(g_cam_theta))
            right_dir = glm.vec3(np.sin(g_cam_theta - np.pi / 2), 0.0, np.cos(g_cam_theta- np.pi / 2))
            g_cam_center -= (front_dir * dy * pan_sens) - (right_dir * dx * pan_sens)
        elif g_z_is_pressed:
            zoom_sens = 0.01
            g_cam_r += dy * zoom_sens
            g_cam_r = max(0.1, g_cam_r)
        else:
            orbit_sens = 0.01
            g_cam_theta -= dx * orbit_sens
            g_cam_phi += dy * orbit_sens
            g_cam_phi = max(-np.pi / 2 + 0.0001, min(g_cam_phi, np.pi / 2 - 0.0001))
    g_mouse_x_pos = xpos
    g_mouse_y_pos = ypos

def framebuffer_size_callback(window, width, height):
    global g_P, g_cam_r
    glViewport(0, 0, width, height)
    if height == 0: height = 1
    per_height = 10.
    per_width = per_height * width/height
    per_as = per_width/per_height
    g_P = glm.perspective(45, per_as, 0.05, 2* g_cam_r)

def prepare_vao_frame():
    vertices = glm.array(glm.float32,
         -5.0, 0.0, 0.0,  1.0, 0.0, 0.0, 
         5.0, 0.0, 0.0,  1.0, 0.0, 0.0, 
         0.0, 0.0, -5.0,  0.0, 0.0, 1.0, 
         0.0, 0.0, 5.0,  0.0, 0.0, 1.0, 
         0.0, -5.0, 0.0,  0.0, 1.0, 0.0, 
         0.0, 5.0, 0.0,  0.0, 1.0, 0.0, 
    )
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)
    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)
    return VAO

def draw_frame(vao, MVP, loc_MVP):
    glBindVertexArray(vao)
    glUniformMatrix4fv(loc_MVP, 1, GL_FALSE, glm.value_ptr(MVP))
    glDrawArrays(GL_LINES, 0, 6)

def draw_grid(vao, MVP, loc_MVP):
    glBindVertexArray(vao)
    scale = 5
    MVP_ = MVP * glm.scale(glm.vec3(scale,scale,scale)) 
    for q in range(-8,7):
        for w in range(-7,8):
            MVP_grid = MVP_ * glm.translate(glm.vec3(1*q/scale, 0, 1*w/scale))
            glUniformMatrix4fv(loc_MVP, 1, GL_FALSE, glm.value_ptr(MVP_grid))
            glDrawArrays(GL_LINES, 0, 4)

# # 이전 답변에서 해결했던 Centering이 적용된 안전한 OBJ 로더
# def load_obj(filename):
#     vertices = []
#     faces = []
#     with open(filename, 'r') as f:
#         for line in f:
#             parts = line.strip().split()
#             if not parts: continue
#             if parts[0] == 'v':
#                 vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
#             elif parts[0] == 'f':
#                 face_indices = []
#                 for v in parts[1:]:
#                     idx = int(v.split('/')[0]) - 1 
#                     face_indices.append(idx)
#                 for i in range(1, len(face_indices) - 1):
#                     faces.append([face_indices[0], face_indices[i], face_indices[i+1]])
    
#     vertices = np.array(vertices, dtype=np.float32)
#     # 강제 Centering으로 치우침 방지
#     min_coords = np.min(vertices, axis=0)
#     max_coords = np.max(vertices, axis=0)
#     center = (min_coords + max_coords) / 2.0
#     vertices -= center
    
#     return vertices, np.array(faces, dtype=np.uint32)

# def prepare_vao_obj(vertices, faces):
#     vertex_data = []
#     for face in faces:
#         p0, p1, p2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
#         normal = np.cross(p1 - p0, p2 - p0)
#         norm_length = np.linalg.norm(normal)
#         normal = normal / norm_length if norm_length > 0 else np.array([0.0, 0.0, 1.0])
#         for idx in face:
#             vertex_data.extend(vertices[idx])
#             vertex_data.extend(normal)

#     v_array = glm.array(glm.float32, *vertex_data)
#     VAO = glGenVertexArrays(1)
#     glBindVertexArray(VAO)
#     VBO = glGenBuffers(1)
#     glBindBuffer(GL_ARRAY_BUFFER, VBO)
#     glBufferData(GL_ARRAY_BUFFER, v_array.nbytes, v_array.ptr, GL_STATIC_DRAW)
    
#     glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
#     glEnableVertexAttribArray(0)
#     glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3 * glm.sizeof(glm.float32)))
#     glEnableVertexAttribArray(1)

#     return VAO, len(faces) * 3
def load_obj(filename):
    vertices = []
    normals = []
    faces = []

    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue

            # 정점 위치 (v x y z)
            if parts[0] == 'v':
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            
            # 법선 벡터 (vn nx ny nz)
            elif parts[0] == 'vn':
                normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
            
            # 면 정보 (f v1/vt1/vn1 v2/vt2/vn2 ...)
            elif parts[0] == 'f':
                face_info = []
                for v_data in parts[1:]:
                    # '/'로 분리 (v, vt, vn)
                    sub_parts = v_data.split('/')
                    v_idx = int(sub_parts[0]) - 1 # OBJ는 1부터 시작하므로 -1
                    
                    vn_idx = -1
                    if len(sub_parts) >= 3 and sub_parts[2]: # vn이 존재하는 경우
                        vn_idx = int(sub_parts[2]) - 1
                    
                    face_info.append((v_idx, vn_idx))
                
                # 다각형을 삼각형으로 분할 (Triangulation)
                for i in range(1, len(face_info) - 1):
                    faces.append([face_info[0], face_info[i], face_info[i+1]])

    # 1. 모델 중심 맞추기 (Centering - 위치 데이터에만 적용)
    v_np = np.array(vertices, dtype=np.float32)
    center = (np.min(v_np, axis=0) + np.max(v_np, axis=0)) / 2.0
    
    # 2. 최종 데이터 생성 (위치와 법선을 짝지음)
    final_vertex_data = []
    for tri in faces:
        for v_idx, vn_idx in tri:
            # 정점 위치 (중심 이동 포함)
            pos = np.array(vertices[v_idx]) - center
            final_vertex_data.extend(pos)
            
            # 법선 벡터 (파일에 vn이 있으면 사용, 없으면 기본값)
            #if vn_idx != -1 and vn_idx < len(normals):
            final_vertex_data.extend(normals[vn_idx])
            # else:
            #     final_vertex_data.extend([0.0, 0.0, 1.0]) # 기본값

    return np.array(final_vertex_data, dtype=np.float32), len(faces) * 3

def prepare_vao_obj(vertex_data, vertex_count):
    v_array = glm.array(glm.float32, *vertex_data)
    
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)
    
    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, v_array.nbytes, v_array.ptr, GL_STATIC_DRAW)
    
    # 포인터 설정 (위치 0: 3 floats, 법선 1: 3 floats)
    stride = 6 * glm.sizeof(glm.float32)
    
    # 0: Position
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, None)
    glEnableVertexAttribArray(0)
    
    # 1: Normal
    normal_offset = ctypes.c_void_p(3 * glm.sizeof(glm.float32))
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, normal_offset)
    glEnableVertexAttribArray(1)

    return VAO, vertex_count

def safe_load_and_prepare_obj(filename):
    obj_path = os.path.join('.', filename)
    if os.path.exists(obj_path):
        obj_vertices, obj_faces = load_obj(obj_path)
        return prepare_vao_obj(obj_vertices, obj_faces)
    else:
        print(f"Error: File not found at {obj_path}")
        sys.exit(1)

# 재귀적으로 트리를 순회하며 렌더링하는 함수들
def draw_node_color(node, VP, loc_MVP, loc_color):
    if node.get_vao() is None: return
    MVP = VP * node.get_global_transform() * node.get_shape_transform()
    glBindVertexArray(node.get_vao())
    glUniformMatrix4fv(loc_MVP, 1, GL_FALSE, glm.value_ptr(MVP))
    glUniform3f(loc_color, *node.get_color())
    glDrawArrays(GL_TRIANGLES, 0, node.get_vertex_count())

def draw_tree_phong(node, VP, loc_MVP, loc_M, loc_object_color):
    if node.get_vao() is not None:
        M = node.get_global_transform() * node.get_shape_transform()
        MVP = VP * M
        glBindVertexArray(node.get_vao())
        glUniformMatrix4fv(loc_MVP, 1, GL_FALSE, glm.value_ptr(MVP))
        glUniformMatrix4fv(loc_M, 1, GL_FALSE, glm.value_ptr(M))
        glUniform3f(loc_object_color, *node.get_color())
        glDrawArrays(GL_TRIANGLES, 0, node.get_vertex_count())
    
    for child in node.children:
        draw_tree_phong(child, VP, loc_MVP, loc_M, loc_object_color)


# --------------------------------------------------------
# 5. 메인 함수
# --------------------------------------------------------
def main():
    global g_P, g_cam_r
    if not glfwInit(): return
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3)
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE)

    window = glfwCreateWindow(1000, 1000, 'Vortex Solar System with Serial Pipes', None, None)
    if not window:
        glfwTerminate()
        return
    glfwMakeContextCurrent(window)

    glfwSetMouseButtonCallback(window, button_callback)
    glfwSetKeyCallback(window, key_callback)
    glfwSetCursorPosCallback(window, cursor_callback)
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback)

    # 셰이더 및 유니폼 로케이션
    shader_program_frame = load_shaders(g_vertex_shader_src_color_attribute, g_fragment_shader_src_color)
    shader_program_sun = load_shaders(g_vertex_shader_src_color_uniform, g_fragment_shader_src_color)
    shader_program_phong = load_shaders(g_vertex_shader_src_phong, g_fragment_shader_src_phong)

    loc_MVP_frame = glGetUniformLocation(shader_program_frame, 'MVP')
    loc_MVP_sun = glGetUniformLocation(shader_program_sun, 'MVP')
    loc_color_sun = glGetUniformLocation(shader_program_sun, 'color')
    
    loc_MVP_phong = glGetUniformLocation(shader_program_phong, 'MVP')
    loc_M_phong = glGetUniformLocation(shader_program_phong, 'M')
    loc_view_pos_phong = glGetUniformLocation(shader_program_phong, 'view_pos')
    loc_light_pos_phong = glGetUniformLocation(shader_program_phong, 'light_pos')
    loc_object_color_phong = glGetUniformLocation(shader_program_phong, 'object_color')

    vao_frame = prepare_vao_frame()

    # 모든 OBJ 로드 (pipe.obj 포함)
    vao_sun, vcnt_sun         = safe_load_and_prepare_obj('sun.obj')
    vao_jupiter, vcnt_jupiter = safe_load_and_prepare_obj('jupiter.obj')
    vao_saturn, vcnt_saturn   = safe_load_and_prepare_obj('saturn.obj')
    vao_earth, vcnt_earth     = safe_load_and_prepare_obj('earth.obj')
    vao_moon, vcnt_moon       = safe_load_and_prepare_obj('moon.obj')
    vao_pipe, vcnt_pipe       = safe_load_and_prepare_obj('pipe.obj')

    # 색상 정의
    color_sun     = glm.vec3(1.0, 0.9, 0.0)
    color_jupiter = glm.vec3(0.8, 0.6, 0.4)
    color_saturn  = glm.vec3(0.9, 0.8, 0.6)
    color_earth   = glm.vec3(0.2, 0.4, 0.8)
    color_moon    = glm.vec3(0.7, 0.7, 0.7)
    color_pipe    = glm.vec3(0.0, 1.0, 0.8) # 형광 하늘색 (자취 효과)

    # -------------------------------------------------------------
    # 계층적 모델링: Base(공전 전용)와 Mesh(렌더링 및 자전 전용) 분리
    # -------------------------------------------------------------
    world_center = Node(None, glm.mat4(), glm.vec3(0), None, 0)

    # 태양 (Vortex: 중심에서 3.0만큼 떨어져서 회전)
    sun_base = Node(world_center, glm.mat4(), glm.vec3(0), None, 0)
    sun_mesh = Node(sun_base, glm.scale(glm.vec3(0.6)), color_sun, vao_sun, vcnt_sun)

    # 행성들 (태양 Base를 중심으로 공전)
    jupiter_base = Node(sun_base, glm.mat4(), glm.vec3(0), None, 0)
    jupiter_mesh = Node(jupiter_base, glm.scale(glm.vec3(0.8)), color_jupiter, vao_jupiter, vcnt_jupiter)

    saturn_base = Node(sun_base, glm.mat4(), glm.vec3(0), None, 0)
    saturn_mesh = Node(saturn_base, glm.scale(glm.vec3(0.7)), color_saturn, vao_saturn, vcnt_saturn)

    earth_base = Node(sun_base, glm.mat4(), glm.vec3(0), None, 0)
    earth_mesh = Node(earth_base, glm.scale(glm.vec3(0.5)), color_earth, vao_earth, vcnt_earth)

    # 달 (지구 Base를 중심으로 공전)
    moon_base = Node(earth_base, glm.mat4(), glm.vec3(0), None, 0)
    moon_mesh = Node(moon_base, glm.scale(glm.vec3(0.3)), color_moon, vao_moon, vcnt_moon)

    # ~ [핵심 로직] 직렬 파이프(자취) 생성 함수 
    # ~ 앞선 파이프를 부모로 삼아 (Serial), 궤적을 거슬러 올라가듯 역회전하는 Transform을 할당합니다.
    def build_serial_trail(parent_base, orbit_radius, num_pipes=25, delta_angle=0.06):
        curr_parent = parent_base
        # 로컬 좌표계에서 부모 중심으로 돌아가서(Translate -r) -> 약간 역회전(Rotate) -> 다시 제자리로(Translate +r)
        # 이 변환이 직렬로 누적되면 원 궤도를 따라 휘어지는 아름다운 꼬리가 생성됩니다.
        step_transform = glm.translate(glm.vec3(-orbit_radius, 0, 0)) * glm.rotate(-delta_angle, glm.vec3(0, 1, 0)) * glm.translate(glm.vec3(orbit_radius, 0, 0))
        
        for i in range(num_pipes):
            # 꼬리로 갈수록 파이프 크기가 작아지게 스케일 조정
            scale_val = 0.25 * (1.0 - (i / num_pipes) * 0.9)
            pipe_node = Node(curr_parent, glm.scale(glm.vec3(scale_val)), color_pipe, vao_pipe, vcnt_pipe)
            pipe_node.set_transform(step_transform)
            curr_parent = pipe_node # 현재 파이프가 다음 파이프의 부모가 됨 (Serial)

    # 목성, 토성, 지구에 직렬 자취 생성 (달 제외)
    build_serial_trail(jupiter_base, 12.0)
    build_serial_trail(saturn_base, 8.5)
    build_serial_trail(earth_base, 5.0)

    while not glfwWindowShouldClose(window):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)

        width, height = glfwGetWindowSize(window)
        if height == 0: height = 1
        per_height = 10.
        per_width = per_height * width/height
        g_P = glm.perspective(45, per_width/per_height, 0.05, 2* g_cam_r)

        eye_x = g_cam_center.x + g_cam_r * np.sin(g_cam_theta) * np.cos(g_cam_phi)
        eye_y = g_cam_center.y + g_cam_r * np.sin(g_cam_phi)
        eye_z = g_cam_center.z + g_cam_r * np.cos(g_cam_phi) * np.cos(g_cam_theta)
        V = glm.lookAt(glm.vec3(eye_x, eye_y, eye_z), g_cam_center, glm.vec3(0,1,0))
        VP = g_P * V

        glUseProgram(shader_program_frame)
        draw_frame(vao_frame, VP * glm.mat4(), loc_MVP_frame)
        draw_grid(vao_frame, VP * glm.mat4(), loc_MVP_frame)

        # --- 애니메이션 업데이트 ---
        t = glfwGetTime()

        # 1. 태양 (Vortex Effect): 중심을 기준으로 공전하며 동시에 자전
        #sun_base.set_transform(glm.rotate(t * 0.2, glm.vec3(0, 1, 0)) * glm.translate(glm.vec3(3.0, 0, 0)))
        sun_mesh.set_transform(glm.rotate(t * 0.8, glm.vec3(0, 1, 0)) * glm.translate(glm.vec3(0.0,t * 0.5, 0.0)))

        # 2. 행성 공전 (Base) 및 자전 (Mesh)
        jupiter_base.set_transform(glm.rotate(t * 0.4, glm.vec3(0, 1, 0)) * glm.translate(glm.vec3(12.0, 0, 0)))
        jupiter_mesh.set_transform(glm.rotate(t * 1.5, glm.vec3(0, 1, 0)))

        saturn_base.set_transform(glm.rotate(t * 0.6, glm.vec3(0, 1, 0)) * glm.translate(glm.vec3(8.5, 0, 0)))
        saturn_mesh.set_transform(glm.rotate(t * 1.5, glm.vec3(0, 1, 0)))

        earth_base.set_transform(glm.rotate(t * 1.0, glm.vec3(0, 1, 0)) * glm.translate(glm.vec3(5.0, 0, 0)))
        earth_mesh.set_transform(glm.rotate(t * 2.0, glm.vec3(0, 1, 0)))

        # 3. 달 공전 및 자전
        moon_base.set_transform(glm.rotate(t * 3.0, glm.vec3(0, 1, 0)) * glm.translate(glm.vec3(1.5, 0, 0)))
        moon_mesh.set_transform(glm.rotate(t * 1.0, glm.vec3(0, 1, 0)))

        # 트리 전체의 Global Transform 갱신
        world_center.update_tree_global_transform()

        # --- 렌더링 ---
        # A. 태양 렌더링 (단색 셰이더)
        glUseProgram(shader_program_sun)
        draw_node_color(sun_mesh, VP, loc_MVP_sun, loc_color_sun)

        # B. 광원 위치 설정 (움직이는 태양의 중심 좌표 추출)
        glUseProgram(shader_program_phong)
        sun_mat = sun_base.get_global_transform()
        sun_pos = glm.vec3(sun_mat[3]) 
        glUniform3f(loc_light_pos_phong, sun_pos.x, sun_pos.y, sun_pos.z)
        glUniform3f(loc_view_pos_phong, eye_x, eye_y, eye_z)

        # C. 행성, 파이프, 달 렌더링 (Phong 셰이더 적용, 트리 순회)
        draw_tree_phong(jupiter_base, VP, loc_MVP_phong, loc_M_phong, loc_object_color_phong)
        draw_tree_phong(saturn_base, VP, loc_MVP_phong, loc_M_phong, loc_object_color_phong)
        draw_tree_phong(earth_base, VP, loc_MVP_phong, loc_M_phong, loc_object_color_phong)

        glfwSwapBuffers(window)
        glfwPollEvents()

    glfwTerminate()

if __name__ == "__main__":
    main()