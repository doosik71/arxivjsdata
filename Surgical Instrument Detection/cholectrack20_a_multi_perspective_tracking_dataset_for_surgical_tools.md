# CholecTrack20: A Multi-Perspective Tracking Dataset for Surgical Tools

Chinedu Innocent Nwoye, Kareem Elgohary, Anvita Srinivas, Fauzan Zaid, Joël L. Lavanchy, Nicolas Padoy (2025)

## 🧩 Problem to Solve

수술 영상에서의 도구 추적(Tool Tracking)은 수술 숙련도 평가, 안전 구역 추정, 인간-기계 협업과 같은 컴퓨터 보조 중재(computer-assisted interventions)를 발전시키는 데 필수적이다. 그러나 현재 이 분야의 AI 적용을 제한하는 핵심적인 문제는 문맥 정보가 풍부한 데이터셋의 부족이다.

기존의 데이터셋들은 지나치게 일반적인 추적 정식화(tracking formalizations)에 의존하고 있으며, 이는 도구가 카메라의 시야(Field of View, FoV)를 벗어나거나 신체 밖으로 나가는 것과 같은 수술 특유의 역동성을 포착하지 못한다. 결과적으로 생성된 궤적은 임상적 관련성이 떨어지며, 실제 수술 환경에서의 유연한 적용이 어렵다. 또한, 기존 방법론들은 연기(smoke), 반사(reflection), 출혈(bleeding)과 같은 수술실의 시각적 난제들에 취약하여 임상 현장으로의 전환에 한계를 보인다.

본 논문의 목표는 이러한 한계를 극복하기 위해 다중 클래스 및 다중 도구 추적을 위한 전문 데이터셋인 **CholecTrack20**을 제안하고, 수술적 관점을 반영한 새로운 추적 프레임워크를 정립하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 기여는 수술 도구의 궤적을 단순히 시각적 가시성에 의존하지 않고, 세 가지 서로 다른 관점에서 정의한 **Multi-Perspective Tracking** 개념을 도입한 것이다.

1. **Multi-Perspective Tracking 정의**: 도구의 궤적을 (1) Intraoperative(수술 전 과정), (2) Intracorporeal(체내 체류), (3) Visibility(카메라 가시성)의 세 가지 관점으로 세분화하여 정의하였다. 이를 통해 임상적으로 훨씬 유의미하고 적응 가능한 도구 궤적 분석이 가능해졌다.
2. **CholecTrack20 데이터셋 구축**: 복강경 담낭 절제술(laparoscopic cholecystectomy) 영상을 기반으로, 20개의 전체 길이 영상(35K 프레임, 65K 도구 인스턴스)에 대해 매우 상세한 어노테이션을 제공한다. 여기에는 공간적 위치뿐 아니라 도구 카테고리, ID, 조작자(operator), 수술 단계(phase), 시각적 난제(visual challenge) 정보가 포함된다.
3. **SOTA 벤치마크 수행**: 최신 객체 검출 및 추적 알고리즘들을 CholecTrack20에 적용하여 성능을 평가함으로써, 현재 기술 수준이 임상 적용을 위해서는 여전히 부족하다는 점을 정량적으로 증명하고 향후 연구 방향을 제시하였다.

## 📎 Related Works

기존의 객체 검출 및 추적 연구는 COCO, KITTI와 같은 일반 데이터셋을 중심으로 발전해 왔으며, 이를 수술 도구 추적에 적용하려는 시도가 있었다. 하지만 수술 환경은 출혈, 연기, 급격한 움직임 및 가변적인 조명으로 인해 일반적인 환경보다 훨씬 복잡하다.

기존 수술 도구 추적 방식은 크게 Single Object Tracking (SOT), 클래스당 도구 하나만 다루는 Multi-Class Tracking (MCT), 또는 모든 도구를 하나의 클래스로 취급하는 Multi-Object Tracking (MOT)으로 나뉜다. 그러나 이러한 방식들은 수술 중 도구가 교체되거나 카메라 시야 밖으로 사라졌다가 다시 나타나는 복잡한 **Multi-Class Multi-Object Tracking (MC-MOT)** 상황을 제대로 처리하지 못한다.

또한, 기존 데이터셋들은 주로 가시성(visibility) 관점의 추적에만 집중하고 있어, 도구가 체내에 머물고 있는지 혹은 수술 전체 과정에서 어떻게 사용되었는지에 대한 세밀한 분석(granularity)이 부족하다는 한계가 있다.

## 🛠️ Methodology

### 전체 시스템 및 데이터 구성

CholecTrack20은 복강경 담낭 절제술 영상에서 추출되었으며, 데이터 누수를 방지하기 위해 영상 단위로 훈련, 검증, 테스트 세트를 $5:1:4$ 비율로 분할하였다. 원본 25 FPS 영상을 어노테이션을 위해 1 FPS로 서브샘플링하여 시간적 일관성을 유지하면서 효율적인 라벨링을 수행하였다.

### 추적 정식화 (Track Formalization)

수술 영상 데이터셋 $\mathcal{D} = \{S_1, S_2, \dots, S_n\}$에서 각 시퀀스 $S_i$는 바운딩 박스 $\mathcal{B} = [B_1, B_2, \dots, B_M]$와 도구 클래스 $\mathcal{C} = \{C_1, C_2, \dots, C_N\}$로 구성된다. 각 도구는 트로카 포트 $\mathcal{P} = [P_1, P_2, \dots, P_M]$와 연결된 조작자에 의해 제어되며, 고유한 트랙 ID를 부여받는다.

추적 문제는 프레임 $t$의 $i$번째 도구와 프레임 $t+1$의 $j$번째 도구 사이의 연관성을 정의하는 연관 행렬 $\mathcal{A}(t)$를 해결하는 것으로 정의된다.
$$\mathcal{A}_{i,j} =
\begin{cases}
1, & \text{if tool } i \text{ at frame } t \text{ is associated with tool } j \text{ at frame } t+1 \\
0, & \text{otherwise}
\end{cases}$$

### Multi-Perspective (MP) Trajectory
본 논문은 도구의 궤적을 다음 세 가지 관점에서 정의한다.

1.  **Intraoperative trajectory (수술 전 과정 궤적)**: 도구가 수술 중 처음 나타난 순간부터 마지막으로 사라질 때까지의 전체 생애 주기를 추적한다. 가림(occlusion)이나 카메라 시야 밖으로 나갔다 돌아오는 경우에도 동일 ID를 유지해야 하므로 Re-identification (Re-ID) 능력이 중요하다. 이는 도구 사용량 모니터링 및 숙련도 평가에 활용된다.
2.  **Intracorporeal trajectory (체내 궤적)**: 도구가 신체 내부로 진입한 순간부터 트로카 포트를 통해 완전히 나갈 때까지를 추적한다. 카메라 시야 밖에 있더라도 다른 도구가 동일한 포트로 진입하거나 도구가 잡고 있던 조직을 놓은 경우 체외로 나간 것으로 간주한다. 이는 수술 워크플로우 이해와 위험 예측에 필수적이다.
3.  **Visibility trajectory (가시성 궤적)**: 도구가 카메라 시야 내에 나타난 순간부터 사라질 때까지를 추적한다. 2초 이내의 짧은 사라짐은 가림으로 간주하여 Re-ID를 수행한다. 이는 실시간 피드백 및 동작 경제성 측정에 유용하다.

### 데이터 어노테이션 및 품질 관리
- **라벨 카테고리**: 7종의 도구(grasper, bipolar, hook, scissors, clipper, irrigator, specimen bag), 4종의 조작자(MSLH, MSRH, ASRH, NULL), 8종의 시각적 난제, 7종의 수술 단계가 정의되었다.
- **품질 보증**: 바운딩 박스의 공간적 중첩은 Jaccard Index로, 카테고리 라벨의 일치도는 Cohen’s Kappa Statistic으로 검증하였다. 모호한 사례는 전문 외과 의사의 중재(Mediation)를 통해 최종 확정하였다.

## 📊 Results

### 도구 검출 (Tool Detection) Benchmark
다양한 SOTA 검출 모델을 벤치마킹한 결과, **YOLOv7**이 $AP_{0.5:0.95}$ 기준 56.1%로 가장 높은 성능을 보였다.
- **도구별 성능**: Hook은 모든 모델에서 높은 검출률을 보였으나, Irrigator와 Specimen bag은 불분명한 경계와 변형 가능한 특성 때문에 낮은 성능을 보였다.
- **시각적 난제**: 모든 모델이 흐릿함(blurring), 트로카 근처(trocar view), 정반사(specular reflection) 상황에서 성능 저하를 겪었다.

### 도구 추적 (Tool Tracking) Benchmark
MOT 방법론들을 적용하여 세 가지 관점별로 성능을 평가한 결과, **Bot-SORT**가 전반적으로 가장 우수한 성능을 기록하였다.
- **관점별 난이도**: Visibility $\rightarrow$ Intraoperative $\rightarrow$ Intracorporeal 순으로 난이도가 높았다.
    - **Visibility**: 가장 쉬우며 Bot-SORT가 44.7% HOTA를 달성하였다.
    - **Intracorporeal**: 가장 어려우며, 최대 HOTA가 27.0%에 그쳤다. 이는 체내 진입/퇴출 판단 요소가 시각적으로 명확하지 않기 때문이다.
- **결과 분석**: 대부분의 모델이 HOTA 45% 미만의 성능을 보였으며, 이는 임상 현장에 적용하기에는 정확도가 턱없이 부족한 수준이다.

## 🧠 Insights & Discussion

### 강점 및 기여
본 연구는 단순히 더 많은 데이터를 제공한 것이 아니라, '수술적 관점'이라는 도메인 지식을 추적 정식화에 녹여냈다는 점에서 큰 강점을 가진다. 특히 Intracorporeal trajectory와 같이 기존 문헌에서 다루지 않았던 복잡한 궤적 정보를 제공함으로써, AI가 수술의 전체 맥락을 이해하도록 유도하는 기반을 마련하였다.

### 한계 및 비판적 해석
실험 결과에서 나타나듯, 현재의 SOTA 추적 모델들은 위치 정보와 외형적 특징(appearance)에만 의존하고 있다. 그러나 수술 도구들은 서로 외형이 매우 유사하며, 빈번한 가림과 시야 이탈이 발생하므로 단순한 특징 기반의 Re-ID로는 한계가 명확하다.

특히 Intracorporeal 추적 성능이 현저히 낮은 점은, 단순히 영상 프레임만 분석하는 것이 아니라 트로카의 위치, 수술 단계의 전이, 조작자의 손 위치와 같은 **문맥적 정보(Context-aware cues)**를 통합적으로 활용하는 새로운 알고리즘 설계가 필요함을 시사한다.

## 📌 TL;DR

본 논문은 복강경 수술 도구 추적을 위한 전문 데이터셋 **CholecTrack20**을 제안한다. 이 데이터셋의 핵심은 추적 궤적을 **가시성(Visibility), 체내 체류(Intracorporeal), 수술 전체 과정(Intraoperative)**의 세 가지 관점으로 정의하여 임상적 유의성을 높인 것이다. SOTA 모델 벤치마크 결과, 현재의 추적 기술은 수술실의 시각적 난제와 복잡한 도구 교체 상황을 처리하기에 부족하며(HOTA < 45%), 향후 외형적 특징을 넘어선 문맥 인식 기반의 추적 알고리즘 연구가 필수적임을 보여준다.
