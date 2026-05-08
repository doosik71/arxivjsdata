# CholecTrack20: A Multi-Perspective Tracking Dataset for Surgical Tools

Chinedu Innocent Nwoye, Kareem Elgohary, Anvita Srinivas, Fauzan Zaid, Joël L. Lavanchy, Nicolas Padoy (2025)

## 🧩 Problem to Solve

수술 영상에서의 도구 추적(Tool Tracking)은 수술 숙련도 평가, 안전 구역 추정, 인간-기계 협업과 같은 컴퓨터 보조 중재(computer-assisted interventions)를 발전시키는 데 필수적이다. 그러나 현재 이 분야의 AI 적용을 제한하는 주요 원인은 문맥 정보가 풍부한 데이터셋의 부족이다.

기존의 데이터셋들은 너무 일반적인 추적 정식화(tracking formalizations)에 의존하고 있어, 도구가 카메라의 시야(Field of View, FoV)를 벗어나거나 신체 밖으로 나가는 것과 같은 수술 특유의 동역학을 포착하지 못한다. 이로 인해 임상적으로 관련성이 낮은 궤적이 생성되며, 실제 수술 환경에 적용하기 위한 유연성이 부족하다. 또한, 기존 방법론들은 연기(smoke), 반사(reflection), 출혈(bleeding)과 같은 수술실의 시각적 난제들에 취약하여 임상 적용에 한계를 보인다.

본 논문의 목표는 이러한 문제를 해결하기 위해 다중 클래스 및 다중 도구 추적을 위한 전문 데이터셋인 **CholecTrack20**을 제안하고, 세 가지 서로 다른 관점(Perspective)에서의 추적 정식화를 통해 임상적으로 의미 있는 도구 궤적을 정의하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 기여는 수술 도구 추적을 단순히 '보이는 것'을 쫓는 것에서 벗어나, 수술의 흐름과 신체 구조를 고려한 **Multi-Perspective Tracking** 개념을 도입한 것이다.

1. **Multi-Perspective Tracking 정의**: 도구의 궤적을 (1) Intraoperative, (2) Intracorporeal, (3) Visibility라는 세 가지 관점에서 정의하여, 용도에 따라 유연하게 선택 가능한 궤적 데이터를 제공한다.
2. **CholecTrack20 데이터셋 구축**: 복강경 담낭 절제술(laparoscopic cholecystectomy) 영상을 기반으로, 공간적 위치뿐만 아니라 도구 카테고리, ID, 조작자(operator), 수술 단계(phase), 시각적 난제(visual challenge)까지 포함된 상세한 어노테이션을 제공한다.
3. **벤치마크 분석**: 최신 객체 탐지 및 추적 알고리즘을 통해 데이터셋의 유효성을 검증하고, 현재의 SOTA 모델들이 임상 적용 수준의 정확도에 도달하지 못했음을 정량적으로 분석하여 향후 연구 방향을 제시한다.

## 📎 Related Works

기존의 도구 추적 연구는 크게 단일 객체 추적(SOT), 클래스당 하나의 도구만 추적하는 다중 클래스 추적(MCT), 혹은 모든 도구를 하나의 클래스로 처리하는 다중 객체 추적(MOT)에 집중되어 왔다. 하지만 수술 환경은 여러 클래스의 도구들이 동시다발적으로 상호작용하는 **Multi-Class Multi-Object Tracking (MC-MOT)** 환경이며, 이는 훨씬 더 복잡한 문제를 야기한다.

기존의 수술 추적 데이터셋들은 대부분 가시성(visibility) 중심의 추적만을 제공하며, 도구가 교체되거나 시야 밖으로 나갔다 돌아오는 상황을 적절히 처리하지 못한다. 또한, 전자기적/광학적 추적 방법은 정확하지만 침습적이거나 비용이 많이 들며, 이미지 기반 접근 방식은 정체성 단편화(identity fragmentation)나 ID 스위치(identity switch) 문제로 인해 낮은 정확도를 보인다.

## 🛠️ Methodology

### 1. 데이터 수집 및 전처리

CholecTrack20은 공개 데이터셋인 Cholec80과 CholecT50의 원본 영상을 기반으로 하며, 수술의 복잡성과 도구의 다양성을 대표하는 20개의 전체 길이 영상을 선정하였다. 어노테이션 효율성을 위해 25 FPS 영상을 1 FPS로 서브샘플링하여 총 35,000프레임 이상, 65,000개의 도구 인스턴스를 추출하였다.

### 2. 추적 정식화 (Track Formalization)

수술 영상 데이터셋 $D = \{S_1, S_2, \dots, S_n\}$에서 각 시퀀스 $S_i$는 도구 위치를 나타내는 바운딩 박스 $B$와 클래스 $C$로 구성된다. 각 도구는 고유한 트랙 ID를 가지며, 이는 시각적 단서(클래스 $c \in C$ 및 위치 $b \in B$)와 문맥적 단서(트로카 포트 $p \in P$와 연결된 외과 의사의 손 위치)를 기반으로 할당된다.

추적 문제는 다음과 같은 연관 행렬(association matrix) $A(t)$를 해결하는 것으로 정의된다.
$$A_{i,j} = \begin{cases} 1, & \text{if the } i\text{-th tool in frame } t \text{ is associated with the } j\text{-th tool in frame } t+1 \\ 0, & \text{otherwise} \end{cases}$$
이를 통해 최종적으로 각 ID별 도구 궤적 $T = \{T_1, T_2, \dots, T_K\}$를 생성한다.

### 3. Multi-Perspective (MP) Trajectory

본 논문의 핵심인 세 가지 추적 관점은 다음과 같다.

- **Intraoperative trajectory (수술 내 궤적)**: 도구가 환자의 몸에 처음 나타난 순간부터 마지막으로 사라질 때까지의 전체 생애 주기를 추적한다. 가려짐(occlusion)이나 시야 밖으로 나갔다 돌아오는 경우에도 동일 ID를 유지하는 Re-identification이 필요하며, 도구 사용량 모니터링 및 숙련도 평가에 활용된다.
- **Intracorporeal trajectory (체내 궤적)**: 도구가 트로카 포트를 통해 체내로 진입한 순간부터 체외로 나가는 순간까지를 추적한다. 카메라 시야 밖에서 발생하더라도 다른 도구가 동일 포트로 진입하거나 기존 도구가 파지물을 놓는 등의 임상적 근거를 통해 종료 시점을 추론한다. 수술 워크플로우 이해와 리스크 추정에 중요하다.
- **Visibility trajectory (가시성 궤적)**: 도구가 카메라 시야(FoV)에 나타난 순간부터 사라진 순간까지를 추적한다. 2초 이내의 짧은 사라짐은 Re-ID를 통해 동일 ID로 유지한다. 실시간 피드백 및 동작 경제성(economy of motion) 측정에 적합하다.

### 4. 어노테이션 상세

- **도구 카테고리 (7종)**: Cold grasper, Bipolar grasper, Monopolar hook, Monopolar scissors, Clipper, Irrigator, Specimen bag.
- **조작자 (4종)**: 주외과 의사 왼손(MSLH), 주외과 의사 오른손(MSRH), 보조 외과 의사 오른손(ASRH), Null.
- **시각적 난제 (8종)**: Blurring, Bleeding, Lens fouling, Crowded scene, Occlusion, Smoke, Specular reflection, Trocar view.
- **수술 단계 (7단계)**: Preparation, Calot triangle dissection, Gallbladder dissection, Clipping & cutting, Gallbladder packaging, Cleaning and coagulation, Gallbladder extraction.

## 📊 Results

### 1. 도구 탐지 (Tool Detection) 벤치마크

다양한 SOTA 탐지 모델(Faster-RCNN, YOLOv7, v8, v9, v10 등)을 평가한 결과, **YOLOv7**이 $\text{AP}_{0.5:0.95}$ 기준 56.1%로 가장 높은 성능을 보였다.

- **클래스별 결과**: Hook은 가장 높은 탐지율(74.7%~96.0%)을 보인 반면, Irrigator와 Specimen bag은 경계가 불분명하거나 변형이 심해 탐지가 어려웠다.
- **시각적 난제 영향**: 모든 모델에서 Blur, Trocar view, Specular reflection 상황에서 성능이 크게 저하되었다.

### 2. 도구 추적 (Tool Tracking) 벤치마크

OCSORT, ByteTrack, Bot-SORT, SMILETrack 등을 평가한 결과, **Bot-SORT**가 전반적으로 가장 우수한 성능을 기록하였다.

- **관점별 난이도**: $\text{Visibility} > \text{Intraoperative} > \text{Intracorporeal}$ 순으로 추적이 쉬웠다.
  - **Visibility**: Bot-SORT가 HOTA 44.7%로 가장 높은 성능을 보였다. 시각적 단서에 의존하는 딥러닝 모델 특성상 가장 유리한 시나리오이다.
  - **Intracorporeal**: 가장 난이도가 높았으며, Bot-SORT 기준 HOTA 27.0%에 그쳤다. 체내 진입/퇴출 판단을 위한 시각적 근거가 부족하기 때문이다.
  - **Intraoperative**: 중간 정도의 난이도를 보였으며, 도구 카테고리 특징이 ID 유지에 도움을 주었다.

### 3. 기타 분석

- **도구별**: Grasper가 가장 빈번하게 사용되며 추적 정확도도 높았으나, Specimen bag은 형태 변형과 오염으로 인해 추적이 매우 어려웠다.
- **단계별**: Clipping & cutting 단계가 활동이 제한적이고 선형적이라 가장 추적이 쉬웠으며, 담낭 박리(dissection) 단계는 복잡한 움직임으로 인해 난이도가 높았다.

## 🧠 Insights & Discussion

본 연구는 SOTA 추적 모델들을 적용했음에도 불구하고, HOTA 지표가 45% 미만에 머물러 있다는 점을 발견하였다. 이는 현재의 기술 수준이 실제 임상 현장에 적용(Clinical Translation)되기에는 턱없이 부족함을 의미한다.

특히, 단순한 위치 정보나 외형(Appearance) 특징만으로는 유사한 외형을 가진 수술 도구들의 정체성을 유지하는 데 한계가 있다. 연기나 출혈과 같은 시각적 난제가 발생했을 때 ID를 유지하는 Re-ID 성능이 급격히 떨어지는데, 이는 단순한 시각 정보 이상의 **문맥 인식(Context-aware)** 방법론이 필요함을 시사한다.

또한, Intracorporeal 궤적의 낮은 성능은 카메라 시야 밖의 상태를 추론하기 위한 도메인 지식이나 이력 정보(History)를 활용한 새로운 알고리즘의 필요성을 뒷받침한다. CholecTrack20은 이러한 고도화된 알고리즘을 개발하고 검증할 수 있는 풍부한 정답지(Ground Truth)를 제공한다는 점에서 큰 가치가 있다.

## 📌 TL;DR

본 논문은 수술 도구 추적의 임상적 실용성을 높이기 위해 **Intraoperative, Intracorporeal, Visibility**라는 세 가지 관점의 궤적을 정의한 **CholecTrack20** 데이터셋을 제안한다. 20편의 전체 수술 영상에 대해 도구 종류, 조작자, 수술 단계, 시각적 난제 등을 상세히 어노테이션 하였다. 벤치마크 결과, 최신 SOTA 모델(Bot-SORT 등)조차 임상 적용 수준의 정확도에 도달하지 못했음이 밝혀졌으며, 이는 향후 수술 도구 추적 연구가 단순 시각 특징을 넘어 문맥 기반의 Re-ID 및 도메인 지식 결합 방향으로 나아가야 함을 시사한다.
