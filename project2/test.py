from OpenGL.GL import *
from glfw.GLFW import *
import glm
import ctypes
import numpy as np
import os
import sys

# --------------------------------------------------------
# 1. 카메라 및 전역 변수 (main.py에서 가져옴)
# --------------------------------------------------------
# Orbit -> 구면좌표계 사용 
# eye point 위치와 up vector 표현가능
g_cam_r = 15.0  # 태양계 전체를 보기 위해 거리를 약간 늘렸습니다.
g_cam_theta = np.radians(45)
g_cam_phi = np.radians(45)

# fittable viewport 사용
g_P = glm.mat4()
g_height = 1000
g_width = 1000

# centor point 위치
g_cam_center = glm.vec3(0.0,0.0,0.0)
# 마우스
g_mouse_is_dragged = False
g_mouse_x_pos = 0.0
g_mouse_y_pos = 0.0
# 키보드 입력
g_z_is_pressed = False
g_x_is_pressed = False


# --------------------------------------------------------
# 2. 셰이더 소스 코드 (1-hierarchical.py & 4-all-components-phong-facenorm.py)
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
    // 3D points in homogeneous coordinates
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
    // 3D points in homogeneous coordinates
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

# --- (3) 조명을 받는 물체(행성, 달)를 위한 Phong 셰이더 ---
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
in vec3 vout_normal;  // interpolated normal

out vec4 FragColor;

uniform vec3 view_pos;
uniform vec3 light_pos;     // 태양의 위치를 유니폼으로 받아옵니다.
uniform vec3 object_color;  // 각 노드(행성)의 고유 색상입니다.

void main()
{
    // light and material properties
    vec3 light_color = vec3(1,1,1);
    float material_shininess = 32.0;

    // light components
    vec3 light_ambient = 0.1*light_color;
    vec3 light_diffuse = light_color;
    vec3 light_specular = light_color;

    // material components
    vec3 material_ambient = object_color;
    vec3 material_diffuse = object_color;
    vec3 material_specular = vec3(1,1,1);  // for non-metal material

    // ambient
    vec3 ambient = light_ambient * material_ambient;

    // for diffiuse and specular
    vec3 normal = normalize(vout_normal);
    vec3 surface_pos = vout_surface_pos;
    vec3 light_dir = normalize(light_pos - surface_pos);

    // diffuse
    float diff = max(dot(normal, light_dir), 0);
    vec3 diffuse = diff * light_diffuse * material_diffuse;

    // specular
    vec3 view_dir = normalize(view_pos - surface_pos);
    vec3 reflect_dir = reflect(-light_dir, normal);
    float spec = pow( max(dot(view_dir, reflect_dir), 0.0), material_shininess);
    vec3 specular = spec * light_specular * material_specular;

    vec3 color = ambient + diffuse + specular;
    FragColor = vec4(color, 1.);
}
'''

class Node:
    def __init__(self, parent, shape_transform, color, vao, vertex_count):
        # hierarchy
        self.parent = parent
        self.children = []
        if parent is not None:
            parent.children.append(self)

        # transform
        self.transform = glm.mat4()
        self.global_transform = glm.mat4()

        # shape
        self.shape_transform = shape_transform
        self.color = color
        
        # vao 와 count 값을 저장.
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
    # build and compile our shader program
    # ------------------------------------
    
    # vertex shader 
    vertex_shader = glCreateShader(GL_VERTEX_SHADER)    # create an empty shader object
    glShaderSource(vertex_shader, vertex_shader_source) # provide shader source code
    glCompileShader(vertex_shader)                      # compile the shader object
    
    # check for shader compile errors
    success = glGetShaderiv(vertex_shader, GL_COMPILE_STATUS)
    if (not success):
        infoLog = glGetShaderInfoLog(vertex_shader)
        print("ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" + infoLog.decode())
        
    # fragment shader
    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)    # create an empty shader object
    glShaderSource(fragment_shader, fragment_shader_source) # provide shader source code
    glCompileShader(fragment_shader)                        # compile the shader object
    
    # check for shader compile errors
    success = glGetShaderiv(fragment_shader, GL_COMPILE_STATUS)
    if (not success):
        infoLog = glGetShaderInfoLog(fragment_shader)
        print("ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" + infoLog.decode())

    # link shaders
    shader_program = glCreateProgram()               # create an empty program object
    glAttachShader(shader_program, vertex_shader)    # attach the shader objects to the program object
    glAttachShader(shader_program, fragment_shader)
    glLinkProgram(shader_program)                    # link the program object

    # check for linking errors
    success = glGetProgramiv(shader_program, GL_LINK_STATUS)
    if (not success):
        infoLog = glGetProgramInfoLog(shader_program)
        print("ERROR::SHADER::PROGRAM::LINKING_FAILED\n" + infoLog.decode())
        
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)

    return shader_program    # return the shader program

# --- 콜백 함수들 (main.py) ---
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
        # xpos는 parameter로 실시간 들어오는 현재 위치
        dx = xpos - g_mouse_x_pos
        dy = ypos - g_mouse_y_pos
        # glfwGetCursorPos(window)
        if g_x_is_pressed:
            # Pan action
            # xz 축만 이동, 현재 바라보고있는 방향에 대해서 -> theta 필요
            pan_sens = 0.01
            front_dir = glm.vec3(np.sin(g_cam_theta), 0.0, np.cos(g_cam_theta))
            right_dir = glm.vec3(np.sin(g_cam_theta - np.pi / 2), 0.0, np.cos(g_cam_theta- np.pi / 2))
            # right_dir은 + 해줘야지 생각처럼 작동함
            g_cam_center -= (front_dir * dy * pan_sens) - (right_dir * dx * pan_sens)
        elif g_z_is_pressed:
            # Zoom action
            # 현재 방향을 유지하고 거리만 바꾸기 -> r 필요
            # 마우스 양옆 움직임은 반영 X.
            zoom_sens = 0.01
            g_cam_r += dy * zoom_sens
            # r은 음수 값이 될 수 없음.
            # 너무 가까우면 좀 보기 그렇다.
            g_cam_r = max(0.1, g_cam_r)
        else:
            # Orbit action
            orbit_sens = 0.01
            g_cam_theta -= dx * orbit_sens
            g_cam_phi += dy * orbit_sens
            # -90 ~ 90 으로만 제한. 예시에서 그렇게 구현되어있고,
            # 그 이상으로 가버리면 상하 컨트롤이 정반대가 되어서 불편함
            g_cam_phi = max(-np.pi / 2 + 0.0001, min(g_cam_phi, np.pi / 2 - 0.0001))
    g_mouse_x_pos = xpos
    g_mouse_y_pos = ypos

def framebuffer_size_callback(window, width, height):
    global g_P, g_cam_r
    glViewport(0, 0, width, height)
    per_height = 10.
    per_width = per_height * width/height
    per_as = per_width/per_height
    g_P = glm.perspective(45, per_as, 0.05, 2* g_cam_r)

def prepare_vao_frame():
    # prepare vertex data (in main memory)
    # grid 그리는데 사용하기 위해 y를 맨뒤 순서로 바꾸기
    vertices = glm.array(glm.float32,
        # position        # color
         -3.0, 0.0, 0.0,  1.0, 0.0, 0.0, # x-axis start
         3.0, 0.0, 0.0,  1.0, 0.0, 0.0, # x-axis end 
         0.0, 0.0, -3.0,  0.0, 0.0, 1.0, # z-axis start
         0.0, 0.0, 3.0,  0.0, 0.0, 1.0, # z-axis end
         0.0, -3.0, 0.0,  0.0, 1.0, 0.0, # y-axis start
         0.0, 3.0, 0.0,  0.0, 1.0, 0.0, # y-axis end 
    )

    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def draw_frame(vao, MVP, loc_MVP):
    glBindVertexArray(vao)
    glUniformMatrix4fv(loc_MVP, 1, GL_FALSE, glm.value_ptr(MVP))
    glDrawArrays(GL_LINES, 0, 6)

def draw_grid(vao, MVP, loc_MVP):
    glBindVertexArray(vao)
    scale = 3
    MVP_ = MVP * glm.scale(glm.vec3(scale,scale,scale)) 
    for q in range(-8,7):
        for w in range(-7,8):
            MVP_grid = MVP_ * glm.translate(glm.vec3(1*q/scale, 0, 1*w/scale))
            glUniformMatrix4fv(loc_MVP, 1, GL_FALSE, glm.value_ptr(MVP_grid))
            glDrawArrays(GL_LINES, 0, 4)

# def load_obj(filename):
#     vertices = []
#     faces = []
#     with open(filename, 'r') as f:
#         for line in f:
#             parts = line.strip().split()
#             if not parts:
#                 continue
            
#             # Read vertices
#             if parts[0] == 'v':
#                 vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                
#             # Read faces
#             elif parts[0] == 'f':
#                 face_indices = []
#                 for v in parts[1:]:
#                     # OBJ indices are 1-based, and we only want the vertex index (ignoring uv/normals for now)
#                     idx = int(v.split('/')[0]) - 1 
#                     face_indices.append(idx)
                    
#                 # Triangulate polygons (handles both triangles and quads)
#                 for i in range(1, len(face_indices) - 1):
#                     faces.append([face_indices[0], face_indices[i], face_indices[i+1]])
                    
#     return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.uint32)

def load_obj(filename):
    vertices = []
    faces = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            
            # Read vertices
            if parts[0] == 'v':
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                
            # Read faces
            elif parts[0] == 'f':
                face_indices = []
                for v in parts[1:]:
                    idx = int(v.split('/')[0]) - 1 
                    face_indices.append(idx)
                    
                for i in range(1, len(face_indices) - 1):
                    faces.append([face_indices[0], face_indices[i], face_indices[i+1]])
    

    # centering... obj 파일 위치 자체가 이상하다!
    vertices = np.array(vertices, dtype=np.float32)
    
    # 1. 모델의 바운딩 박스(최소/최대 좌표)를 구하여 중심점(Center) 찾기
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    center = (min_coords + max_coords) / 2.0
    
    # 2. 모든 정점 좌표에서 중심점을 빼서 강제로 원점(0,0,0)으로 이동 (해결법)
    vertices -= center
                    
    return vertices, np.array(faces, dtype=np.uint32)

def prepare_vao_obj(vertices, faces):
    vertex_data = []
    
    # 4-all-components-phong-facenorm.py를 적용하기 위해, 색상 대신 "법선(Normal)"을 계산하여 넣습니다.
    for face in faces:
        p0 = vertices[face[0]]
        p1 = vertices[face[1]]
        p2 = vertices[face[2]]
        
        # Face 법선 벡터 계산
        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        norm_length = np.linalg.norm(normal)
        if norm_length > 0:
            normal = normal / norm_length
        else:
            normal = np.array([0.0, 0.0, 1.0])
            
        for idx in face:
            pos = vertices[idx]
            vertex_data.extend(pos)
            vertex_data.extend(normal) # 색상 대신 법선(Normal)을 VBO에 추가

    v_array = glm.array(glm.float32, *vertex_data)

    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)

    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, v_array.nbytes, v_array.ptr, GL_STATIC_DRAW)

    # Position attribute (location 0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # Normal attribute (location 1) - Phong 셰이더용
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3 * glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO, len(faces) * 3

def safe_load_and_prepare_obj(filename):
    obj_path = os.path.join('.', filename)
    if os.path.exists(obj_path):
        obj_vertices, obj_faces = load_obj(obj_path)
        return prepare_vao_obj(obj_vertices, obj_faces)
    else:
        print(f"Error: File not found at {obj_path}")
        sys.exit(1)

# 태양용 Draw (조명 없음, 단일 색상)
def draw_node_color(node, VP, loc_MVP, loc_color):
    vao = node.get_vao()
    vertex_count = node.get_vertex_count()
    color = node.get_color()
    
    MVP = VP * node.get_global_transform() * node.get_shape_transform()
    
    glBindVertexArray(vao)
    glUniformMatrix4fv(loc_MVP, 1, GL_FALSE, glm.value_ptr(MVP))
    glUniform3f(loc_color, color.r, color.g, color.b)
    glDrawArrays(GL_TRIANGLES, 0, vertex_count)

# 행성 및 달용 Draw (Phong Shading)
def draw_node_phong(node, VP, loc_MVP, loc_M, loc_object_color):
    vao = node.get_vao()
    vertex_count = node.get_vertex_count()
    color = node.get_color()
    
    M = node.get_global_transform() * node.get_shape_transform()
    MVP = VP * M
    
    glBindVertexArray(vao)
    glUniformMatrix4fv(loc_MVP, 1, GL_FALSE, glm.value_ptr(MVP))
    glUniformMatrix4fv(loc_M, 1, GL_FALSE, glm.value_ptr(M))
    glUniform3f(loc_object_color, color.r, color.g, color.b)
    glDrawArrays(GL_TRIANGLES, 0, vertex_count)


# --------------------------------------------------------
# 5. 메인 함수
# --------------------------------------------------------
def main():
    global g_P, g_cam_r
    # initialize glfw
    if not glfwInit():
        return
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3)   # OpenGL 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3)
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)  # Do not allow legacy OpenGl API calls
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE) # for macOS

    # create a window and OpenGL context
    window = glfwCreateWindow(1000, 1000, 'Solar System with Phong', None, None)
    if not window:
        glfwTerminate()
        return
    glfwMakeContextCurrent(window)

    # register event callbacks
    glfwSetMouseButtonCallback(window, button_callback)
    glfwSetKeyCallback(window, key_callback)
    glfwSetCursorPosCallback(window, cursor_callback)
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback)

    # 1. 셰이더 로드
    shader_program_frame = load_shaders(g_vertex_shader_src_color_attribute, g_fragment_shader_src_color)
    shader_program_sun = load_shaders(g_vertex_shader_src_color_uniform, g_fragment_shader_src_color)
    shader_program_phong = load_shaders(g_vertex_shader_src_phong, g_fragment_shader_src_phong)

    # 2. 유니폼 로케이션 획득
    loc_MVP_frame = glGetUniformLocation(shader_program_frame, 'MVP')
    
    loc_MVP_sun = glGetUniformLocation(shader_program_sun, 'MVP')
    loc_color_sun = glGetUniformLocation(shader_program_sun, 'color')
    
    loc_MVP_phong = glGetUniformLocation(shader_program_phong, 'MVP')
    loc_M_phong = glGetUniformLocation(shader_program_phong, 'M')
    loc_view_pos_phong = glGetUniformLocation(shader_program_phong, 'view_pos')
    loc_light_pos_phong = glGetUniformLocation(shader_program_phong, 'light_pos')
    loc_object_color_phong = glGetUniformLocation(shader_program_phong, 'object_color')

    # 3. VAO 준비
    vao_frame = prepare_vao_frame()

    # OBJ 로드 및 VAO 생성
    vao_sun, vcnt_sun         = safe_load_and_prepare_obj('sun.obj')
    vao_jupiter, vcnt_jupiter = safe_load_and_prepare_obj('jupiter.obj')
    vao_saturn, vcnt_saturn   = safe_load_and_prepare_obj('saturn.obj')
    vao_earth, vcnt_earth     = safe_load_and_prepare_obj('earth.obj')
    vao_moon, vcnt_moon       = safe_load_and_prepare_obj('moon.obj')

    # 4. 계층적 모델링 트리(Node) 생성
    color_sun     = glm.vec3(1.0, 0.9, 0.0) # 노랑
    color_jupiter = glm.vec3(0.8, 0.6, 0.4) # 갈색 계열
    color_saturn  = glm.vec3(0.9, 0.8, 0.6) # 옅은 갈색
    color_earth   = glm.vec3(0.2, 0.4, 0.8) # 파랑
    color_moon    = glm.vec3(0.7, 0.7, 0.7) # 회색

    # Level 1
    sun_node = Node(None, glm.scale(glm.vec3(0.6, 0.6, 0.6)), color_sun, vao_sun, vcnt_sun)
    
    # Level 2 (모두 Sun의 자식)
    jupiter_node = Node(sun_node, glm.scale(glm.vec3(0.8, 0.8, 0.8)), color_jupiter, vao_jupiter, vcnt_jupiter)
    saturn_node  = Node(sun_node, glm.scale(glm.vec3(0.7, 0.7, 0.7)), color_saturn, vao_saturn, vcnt_saturn)
    earth_node   = Node(sun_node, glm.scale(glm.vec3(0.5, 0.5, 0.5)), color_earth, vao_earth, vcnt_earth)
    
    # Level 3 (Earth의 자식)
    moon_node    = Node(earth_node, glm.scale(glm.vec3(0.3, 0.3, 0.3)), color_moon, vao_moon, vcnt_moon)

    # initialize projection matrix
    per_height = 10.
    per_width = per_height
    per_as = per_width/per_height
    g_P = glm.perspective(45, per_as, 0.05, 2 * g_cam_r)

    # loop until the user closes the window
    while not glfwWindowShouldClose(window):
        # render
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)

        width, height = glfwGetWindowSize(window)
        per_height = 10.
        per_width = per_height * width/height
        per_as = per_width/per_height
        g_P = glm.perspective(45, per_as, 0.05, 2* g_cam_r)

        # view matrix (main.py 방식)
        eye_x = g_cam_center.x + g_cam_r * np.sin(g_cam_theta) * np.cos(g_cam_phi)
        eye_y = g_cam_center.y + g_cam_r * np.sin(g_cam_phi)
        eye_z = g_cam_center.z + g_cam_r * np.cos(g_cam_phi) * np.cos(g_cam_theta)
        
        V = glm.lookAt(
            glm.vec3(eye_x, eye_y, eye_z),
            g_cam_center, 
            glm.vec3(0,1,0)
        )
        VP = g_P * V

        # --- A. 프레임 및 그리드 렌더링 ---
        glUseProgram(shader_program_frame)
        draw_frame(vao_frame, VP * glm.mat4(), loc_MVP_frame)
        draw_grid(vao_frame, VP * glm.mat4(), loc_MVP_frame)

        # --- B. 계층적 애니메이션 업데이트 ---
        t = glfwGetTime()

        # 태양 자전
        sun_node.set_transform(glm.rotate(t * 0.5, glm.vec3(0, 1, 0)))

        # 목성 공전 및 자전 (거리 10.0)
        jupiter_node.set_transform(glm.rotate(t * 0.4, glm.vec3(0, 1, 0)) * glm.translate(glm.vec3(10.0, 0, 0)) * glm.rotate(t * 1.5, glm.vec3(0, 1, 0)))
        
        # 토성 공전 및 자전 (거리 7.0)
        saturn_node.set_transform(glm.rotate(t * 0.6, glm.vec3(0, 1, 0)) * glm.translate(glm.vec3(7.0, 0, 0)) * glm.rotate(t * 1.5, glm.vec3(0, 1, 0)))

        # 지구 공전 및 자전 (거리 4.0)
        earth_node.set_transform(glm.rotate(t * 1.0, glm.vec3(0, 1, 0)) * glm.translate(glm.vec3(10.0, 0, 0)) * glm.rotate(t * 2.0, glm.vec3(0, 1, 0)))

        # 달 지구 중심 공전 및 자전 (지구 기준 거리 2.0)
        moon_node.set_transform(glm.rotate(t * 3.0, glm.vec3(0, 1, 0)) * glm.translate(glm.vec3(5.0, 0, 0)) * glm.rotate(t * 1.0, glm.vec3(0, 1, 0)))

        # recursively update global transformations of all nodes
        sun_node.update_tree_global_transform()


        # --- C. 태양 렌더링 (조명 무시, 단색 셰이더) ---
        glUseProgram(shader_program_sun)
        draw_node_color(sun_node, VP, loc_MVP_sun, loc_color_sun)

        # --- D. 행성 및 달 렌더링 (Phong 셰이더 적용) ---
        glUseProgram(shader_program_phong)

        # 태양의 현재 글로벌 위치를 추출하여 광원(light_pos)으로 설정
        sun_mat = sun_node.get_global_transform()
        sun_pos = glm.vec3(sun_mat[3]) # 행렬의 마지막 열이 Translation(위치) 성분입니다.
        
        glUniform3f(loc_light_pos_phong, sun_pos.x, sun_pos.y, sun_pos.z)
        glUniform3f(loc_view_pos_phong, eye_x, eye_y, eye_z)

        # Phong 셰이더로 행성, 달 그리기
        draw_node_phong(jupiter_node, VP, loc_MVP_phong, loc_M_phong, loc_object_color_phong)
        draw_node_phong(saturn_node, VP, loc_MVP_phong, loc_M_phong, loc_object_color_phong)
        draw_node_phong(earth_node, VP, loc_MVP_phong, loc_M_phong, loc_object_color_phong)
        draw_node_phong(moon_node, VP, loc_MVP_phong, loc_M_phong, loc_object_color_phong)

        # swap front and back buffers
        glfwSwapBuffers(window)
        # poll events
        glfwPollEvents()

    # terminate glfw
    glfwTerminate()

if __name__ == "__main__":
    main()