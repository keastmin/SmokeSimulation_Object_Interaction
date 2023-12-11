# SmokeSimulation_Object_Interaction
연기와 오브젝트 간의 상호작용 시뮬레이션


**시뮬레이션 설명**  
Jos Stam의 논문 기반으로 작성된 Real Time Smoke Simulation을 CUDA를 이용하여 구현하였으며 그 연기 시뮬레이션에 구체를 던졌을 때 상호작용이 어떻게 일어나는지 확인 할 수 있다.

  
**함수 설명**  
1. 연기

add_source: 초기 속도와 밀도를 현재의 속도와 밀도에 추가  

diffuse: 유체의 점성에 의한 속도의 확산을 처리  
    -  gauss seidel 대신 병렬 처리에 용이한 red black gauss seidel이 사용되었다.  

project: 속도 필드를 발산성이 없는 상태로 만듦  
    -  압력을 포아송 방정식으로 구하고, 이 압력을 사용하여 속도를 수정한다.  
    -  이 방정식을 해결하기 위해 red black gauss seidel이 사용된다.  

advect: 유체의 속도에 의해 속도 자체가 어떻게 이동되는지 계산  
    -  이 단계에서 semi-Lagrangian이 사용된다.  

set_bnd: 시뮬레이션의 경계 조건을 정의하여 연기의 흐름을 제한함  

2. 구체

클래스 형식으로 구체의 크기, 현재 위치, 이전 위치, 방향, 속도, 그리드와 충돌이 일어난 셀, 진행 방향과 충돌이 일어난 셀을 저장하여 메인 함수에서 솔버에 경계 조건을 전달한다.  

check_collision: 그리드와 충돌이 일어난 셀 중 연기의 흐름을 변화할 셀, 연기가 구체를 통과하지 못하도록 경계 조건을 정의한 셀, 구체의 내부에 해당하는 셀을 정의함  

(main 소스에 있는)collision_direction_force: 구체에 저장된 방향과 속도를 기반으로 각 방향에 해당하는 소스를 전달함  
  


**사용 API**  
- OpenGL  
- GLEW  
- GLFW  
- GLM  
- CUDA 12.2     
  

**시뮬레이션 환경**  
- CPU : intel i7-10 10700K  
- GPU : RTX 3070 BLACK EDITION OC D6 8GB  
- RAM : samsung DDR4-3200(8GB) x 2  
  

**참고 논문**  
- "Real-time fluid dynamics for games" by jos stam  
- "Stable fluids" by jos stam  

**실행하기 위한 환경**  
- visual studio2022
- cuda 12.2  

**컨트롤**  
- WASD : 카메라 이동  
- 마우스 이동 : 카메라 방향 전환  
- 마우스 좌클릭 : 오브젝트 상호작용    
- 키보드 1번 : 연기 표시  
- 키보드 2번 : 속도 표시  
- Z : 소스항 추가 및 외력 추가
- M : 모드 변경
- C : 연기 초기화  
  

**추가 사항**  
- main 코드의 add_force_source 함수를 통해 외력의 방향을 수정할 수 있음
- main 코드의 mouse_callback 함수를 수정하여 좌클릭 시 구체가 발사되는 지점을 정할 수 있음
- density.cu에서 그리드의 충돌 영역에 대해 그리기 가능  
  
**결과**  
![Image Alt Text](https://github.com/keastmin/SmokeSimulation_ThrowSphere/blob/main/image/3DBullet.gif)  
![Image Alt Text](https://github.com/keastmin/SmokeSimulation_ThrowSphere/blob/main/image/3DBullet_Coll.gif)  
![Image Alt Text](https://github.com/keastmin/SmokeSimulation_ThrowSphere/blob/main/image/forward3DBullet.gif)  
