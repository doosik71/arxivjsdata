# Video-based Surgical Skills Assessment using Long term Tool Tracking

Mona Fathollahi, Mohammad Hasan Sarhan, Ramon Pena, Lela DiMonte, Anshu Gupta, Aishani Ataliwala, Jocelyn Barker (2022)

## 🧩 Problem to Solve

본 논문은 수술 집도의의 기술적 숙련도를 비디오 기반으로 자동 평가하는 문제를 해결하고자 한다. 외과 의사가 수술 기술을 연마하기 위해서는 전문가의 피드백이 필수적이지만, 현재의 비디오 기반 평가는 전문가가 직접 영상을 검토하는 수동 리뷰 방식에 의존하고 있다. 이러한 방식은 시간 소모가 매우 크며, 많은 수술 케이스에 걸쳐 집도의의 발전 과정을 추적하는 데 한계가 있다.

특히, 기존의 자동화 시도들은 로봇 수술 시스템의 키네마틱(Kinematic) 데이터를 활용하여 도구의 궤적을 정확히 측정했지만, 전 세계적으로 대부분의 최소 침습 수술(MIP)은 로봇이 아닌 복강경(Laparoscopic) 방식으로 수행된다. 따라서 별도의 센서나 로봇 시스템 없이 오직 비디오 스트림만으로 도구의 움직임을 추적하고 숙련도를 평가할 수 있는 일반적인 솔루션이 필요하다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 **장기적인 도구 추적(Long-term Tool Tracking)**을 통해 신뢰할 수 있는 움직임 궤적을 생성하고, 이를 기반으로 집도의의 기술 수준을 분류하는 파이프라인을 구축하는 것이다.

주요 기여 사항은 다음과 같다.

1. **ID-switch를 최소화하는 추적 알고리즘**: 수술 도구가 화면 밖으로 나가거나 일시적으로 가려지는 상황에서도 동일한 도구임을 유지할 수 있도록 맞춤형 비용 함수(Cost Function)와 트랙 복구(Track Recovery) 정책을 제안하였다.
2. **Transformer 기반의 숙련도 평가 모델**: 추출된 도구 궤적의 단기 및 장기 패턴을 학습하기 위해 self-attention 메커니즘을 갖춘 Transformer 네트워크를 도입하여 숙련도를 분류하였다.
3. **실제 임상 데이터 검증**: 공개 데이터셋인 Cholec80을 사용하여, 전문가가 평가한 GOALS(Global Operative Assessment of Laparoscopic Skills) 점수와 모델의 예측 결과 간의 상관관계를 검증하였다.

## 📎 Related Works

기존 연구들은 도구의 경로 길이(Path length), 속도, 가속도, 저크(Jerk) 등의 움직임 지표가 외과 의사의 숙련도와 상관관계가 있음을 보여주었다. 그러나 이러한 접근 방식들은 다음과 같은 한계점을 가진다.

- **데이터 획득의 제한**: 주로 로봇 수술의 키네마틱 데이터나 실험실 환경의 특수 센서를 사용하여 3D 궤적을 얻었으며, 일반적인 복강경 비디오만으로는 구현하기 어려웠다.
- **추적의 불안정성**: 컴퓨터 비전 기반의 추적 방법들은 짧은 영상(Short-term)에 최적화되어 있어, 실제 수술과 같은 장시간 영상에서는 ID-switch(추적 대상의 식별자가 바뀌는 현상)가 빈번하게 발생한다.
- **반자동 평가**: 일부 연구는 도구 탐지와 궤적 생성까지는 자동화했으나, 최종 숙련도 판단은 인간이 시각적으로 수행하는 반자동 방식에 머물러 있었다.

## 🛠️ Methodology

### 전체 파이프라인

본 시스템은 **[도구 탐지 $\rightarrow$ 도구 추적 $\rightarrow$ 궤적 추출 $\rightarrow$ 숙련도 분류]** 순서로 구성된다.

### 도구 추적 알고리즘 (Tracking Algorithm)

Tracking-by-detection 방식을 사용하며, YOLOv5를 통해 각 프레임의 도구를 탐지한다. 이후 탐지된 객체를 기존 트랙에 연결하기 위해 Kalman Filter로 위치를 예측하고 Hungarian Algorithm으로 최적의 매칭을 수행한다.

**1. 비용 함수 (Cost Function)**
트랙 $t$와 새로운 탐지 결과 $d$ 사이의 비용을 계산하는 식은 다음과 같다.

$$cost(t, d) = D_{feat}(t, d) + M \cdot \mathbb{1}_{D_{spatial}(t,d) > \lambda_{sp}} + M \cdot \mathbb{1}_{d.DetClass \neq t.classID}$$

- $D_{feat}(t, d)$: Re-identification(Re-ID) 네트워크를 통해 추출된 외형 특징(Appearance feature) 간의 거리이다. 최신 $N$개의 프레임만 사용하는 short-term re-identification을 적용하여 외형 변화에 적응하도록 하였다.
- $D_{spatial}(t, d)$: 탐지된 중심점과 Kalman Filter로 예측된 위치 사이의 거리이다. 임계값 $\lambda_{sp}$를 초과하면 매우 높은 비용 $M$을 부여하여 매칭을 방지한다.
- $d.DetClass \neq t.classID$: 탐지된 도구의 클래스가 기존 트랙의 클래스와 다를 경우 높은 비용 $M$을 부여한다.

**2. 트랙 복구 (Track Recovery)**
탐지 결과가 활성 트랙(Active track)에 할당되지 않은 경우, 비활성 트랙(Inactive track) 중에서 다시 매칭을 시도한다. 이때의 비용 함수는 다음과 같다.

$$cost(t, d) = D_{feat}(t, d) \cdot M \cdot \mathbb{1}_{D_{spatial}(t,d) > \lambda_{sp} \wedge d.DetClass \neq t.classID}$$

이 과정은 도구가 장기에 가려졌다가 다시 나타날 때, 클래스 식별은 불안정할 수 있으나 외형 특징과 공간적 위치를 통해 원래의 ID를 회복하기 위함이다.

### 숙련도 평가 모델 (Skill Assessment)

추출된 2D 궤적 데이터를 사용하여 두 가지 방식으로 숙련도를 분류하였다.

**1. 특징 기반 방식 (Feature-based)**
경로 길이, 속도, 가속도, 저크, 곡률, 굴곡도(Tortuosity) 등 전통적인 모션 메트릭을 직접 계산하여 Random Forest 모델에 입력한다.

**2. 학습 기반 방식 (Learning-based / Transformer)**
도구의 좌표($x, y$)와 바운딩 박스 면적 정보를 입력으로 사용한다.

- **Conv1D 모듈**: 입력 시계열 데이터를 1차원 컨볼루션 층에 통과시켜 원자적 움직임(Atomic motion)을 학습하고 시간 해상도를 낮춘다.
- **Transformer Encoder**: Self-attention 메커니즘을 통해 도구 움직임의 단기 및 장기 의존성을 학습한다. (num\_heads=7, num\_layers=2)
- **출력**: 최종적으로 집도의의 효율성(Efficiency) 수준을 하위(Low) 또는 상위(High) 그룹으로 이진 분류한다.

## 📊 Results

### 실험 설정

- **데이터셋**: Cholec80 데이터셋 중 Calot Triangle Dissection 단계의 마지막 3분 영상을 사용하였다.
- **기준**: 두 명의 전문가가 GOALS 척도를 통해 평가한 효율성 점수의 평균이 3.5점 이상이면 High, 미만이면 Low로 라벨링하였다. (High 51건, Low 29건)

### 추적 모델 성능

제안하는 추적 알고리즘을 최신 기법인 ByteTrack과 비교한 결과는 다음과 같다.

| Method | IDs $\downarrow$ | MOTA $\uparrow$ | Precision $\uparrow$ | Recall $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: |
| ByteTrack | 210 | 89.2% | 95.2% | 94.4% |
| **Proposed** | **87** | 86.6% | **98.1%** | 88.5% |

제안 방법은 MOTA(Multi-Object Tracking Accuracy)는 약간 낮으나, **ID-switch(IDs)를 약 141% 개선**하여 장기적인 추적 안정성을 확보하였다.

### 숙련도 평가 결과

특징 기반 방식과 학습 기반 방식의 성능을 비교 분석하였다.

| Method | Tracking | Precision | Recall | Accuracy | Kappa |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Feature-based | Proposed | 0.68 | 0.52 | 0.65 | 0.30 |
| 1D Convolution | Proposed | 0.83 | 0.75 | 0.74 | 0.45 |
| Transformer | ByteTrack | 0.73 | 0.73 | 0.69 | 0.36 |
| **Transformer** | **Proposed** | **0.88** | **0.84** | **0.83** | **0.63** |

- **결과 분석**: 전통적인 특징 기반 방식은 성능이 매우 낮았다. 이는 3D 키네마틱 데이터와 달리 2D 비디오 기반 궤적에서는 단순 메트릭만으로 숙련도를 구분하기 어렵기 때문이다.
- **최종 성능**: 제안하는 추적 알고리즘과 Transformer 모델을 결합했을 때 가장 높은 정확도(0.83)와 Kappa 계수(0.63)를 기록하였다.

## 🧠 Insights & Discussion

본 연구는 단순한 추적 정확도(MOTA)보다 **ID-switch를 줄이는 것이 숙련도 평가라는 최종 목적에 훨씬 중요하다**는 것을 입증하였다. 도구의 ID가 빈번하게 바뀌면 궤적 데이터에 노이즈가 섞여 모델이 집도의의 일관된 움직임 패턴을 학습할 수 없기 때문이다.

또한, 실제 임상 영상(In-vivo)에서는 도구의 가려짐, 카메라 움직임, 척도의 비표준화 등의 문제가 존재한다. 이러한 환경에서는 사람이 설계한 수동 특징(Handcrafted features)보다 Transformer와 같은 딥러닝 모델이 데이터로부터 직접 유의미한 특징을 추출하는 것이 훨씬 효과적임을 보여주었다.

모델의 Kappa 계수(0.63)가 인간 평가자 간의 일치도(0.41)보다 높게 나타난 점은, 제안된 모델이 인간 수준의 평가 성능을 가졌거나 혹은 더 일관된 기준을 제시할 가능성이 있음을 시사한다.

다만, 본 연구는 Calot Triangle Dissection이라는 특정 수술 단계에만 한정되어 평가되었다는 한계가 있다. 향후 봉합(Suturing)이나 스테이플링(Stapling)과 같은 다른 수술 동작으로의 일반화 가능성을 확인하고, 양손 협응력(Bimanual dexterity)이나 깊이 지각(Depth perception)과 같은 세부 항목 평가로 확장할 필요가 있다.

## 📌 TL;DR

본 논문은 복강경 수술 비디오에서 **ID-switch를 획기적으로 줄인 장기 도구 추적 알고리즘**과 **Transformer 기반의 모션 분석 모델**을 결합하여 집도의의 숙련도를 자동 평가하는 프레임워크를 제안하였다. 실험 결과, 정교한 궤적 추적이 뒷받침된 딥러닝 모델이 전통적인 모션 메트릭 기반 방식보다 훨씬 정확하게 숙련도를 분류함을 확인하였다. 이 연구는 고가의 장비 없이 일반 비디오만으로 외과 의사에게 객관적인 피드백을 제공할 수 있는 확장 가능한 평가 시스템의 가능성을 열어주었다.
