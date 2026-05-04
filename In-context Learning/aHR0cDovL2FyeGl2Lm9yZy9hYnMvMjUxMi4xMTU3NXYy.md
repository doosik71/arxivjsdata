# In-Context Learning for Seismic Data Processing

Fabian Fuchs, Mario Ruben Fernandez, Norman Ettrich, and Janis Keuper (2025)

## 🧩 Problem to Solve

본 논문은 지진파 데이터 처리, 특히 **Seismic Demultiple(다중 반사파 제거)** 과정에서 발생하는 기존 방법론들의 한계를 해결하고자 한다. 지진파 처리의 목적은 가공되지 않은 데이터를 지하 구조 이미지로 변환하는 것이나, 이 과정에서 다음과 같은 문제들이 발생한다.

1. **전통적 방법의 한계**: Radon Transform이나 Surface-Related Multiple Elimination(SRME)과 같은 전통적인 알고리즘은 계산 비용이 매우 높고, 속도 모델링이나 Mute function 설정과 같은 수동의 파라미터 튜닝 과정이 필수적이며, 이는 전문가의 상당한 숙련도를 요구한다.
2. **기존 딥러닝 방법의 한계**: 최근 CNN, GAN, Diffusion 모델 등을 이용한 딥러닝 기반 접근법이 등장했으나, 대부분의 모델이 개별 **CDP(Common Depth Point) gather**를 독립적으로 처리한다. 이로 인해 인접한 gather 간의 결과가 서로 다른 **Lateral Inconsistency(측면 불일치)** 문제가 발생하며, 모델이 블랙박스로 동작하여 사용자가 결과물을 제어할 수 없다는 단점이 있다.

따라서 본 연구의 목표는 인접한 데이터 간의 공간적 상관관계를 활용하여 측면 일관성을 확보하고, 사용자 정의 예시를 통해 모델의 동작을 제어할 수 있는 **In-Context Learning(ICL)** 기반의 지진파 처리 모델인 **ContextSeisNet**을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 자연어 처리(NLP)와 컴퓨터 비전(CV)에서 성공을 거둔 **In-Context Learning(ICL)** 개념을 지진파 데이터 처리에 최초로 도입한 것이다. 

중심적인 설계 직관은 모델이 추론 시점에 특정 작업에 대한 예시(Support Set)를 함께 입력받음으로써, 재학습 없이도 해당 데이터셋의 특성에 맞는 처리 동작을 학습하게 하는 것이다. 이를 통해 인접한 CDP gather들 사이의 공간적 연속성을 강제하여 결과의 일관성을 높이고, 사용자가 제공하는 프롬프트를 통해 딥러닝 모델의 출력을 가이드할 수 있는 제어 가능성(Controllability)을 부여한다.

## 📎 Related Works

1. **전통적 다중 반사파 제거**: 
   - **SRME**: 표면 다중 반사파를 기본 레이패스의 조합으로 표현하여 제거하지만, 계산량이 많고 조밀한 데이터 획득이 필요하다.
   - **Radon Transform (RT)**: NMO-corrected CDP gather에서 기본 반사파(Primaries)와 다중 반사파(Multiples)의 Moveout 차이를 이용하여 분리한다. 하지만 파라미터 튜닝에 민감하며, 단일 reference CDP에서 정의된 Mute function이 전체 서베이 영역에 일반화되지 않는 문제가 있다.
2. **딥러닝 기반 접근법**: 
   - CNN, GAN, Diffusion 모델 등을 이용해 다중 반사파를 제거하려는 시도가 있었으나, 앞서 언급한 것처럼 개별 gather를 독립적으로 처리함으로써 발생하는 측면 불일치 문제를 해결하지 못했다.
3. **In-Context Learning (ICL)**: 
   - LLM에서 시작되어 최근 SegGPT나 UniverSeg 같은 비전 모델로 확장되었다. 본 논문은 이러한 ICL을 지진파 데이터의 공간적 상관관계 유지 및 도메인 적응(Domain Adaptation)을 위해 활용한다.

## 🛠️ Methodology

### 전체 시스템 구조
ContextSeisNet은 개별 gather $X$만 입력받는 일반적인 지도 학습과 달리, 쿼리 이미지 $X$와 함께 공간적으로 연관된 예시 쌍들로 구성된 **Support Set $V$**를 입력으로 받는다.

- **수식적 정의**:
  전통적인 지도 학습이 $f_{\theta}(X[m]) = Y^*_m$ 으로 정의된다면, ICL 기반의 ContextSeisNet은 다음과 같이 정의된다.
  $$f_{ICL_{\theta}}(X[m]|V_m) = Y^*_m$$
  여기서 $V_m = \{ (X[s], f_V(X[s])) : \text{TopS}(\text{sim}(X[m], X[s])) > \tau \}$ 이며, 이는 쿼리 $X[m]$과 유사한 인접 CDP gather들과 그에 대응하는 레이블(전통적 방법인 Radon 등으로 생성 가능)의 집합이다.

### 모델 아키텍처: ContextSeisNet
본 모델은 의료 영상 분할 모델인 **UniverSeg**를 기반으로 하며, U-Net의 Encoder-Decoder 구조를 유지하되 표준 Convolution 블록을 **CrossBlock**으로 대체하였다.

- **CrossBlock의 동작**:
  CrossBlock은 쿼리 특징 맵 $u$와 서포트 특징 맵 $V^{feature}$ 사이의 상호작용을 계산한다.
  1. 쿼리 $u$와 각 서포트 예시 $V^{feature}_s$를 채널 방향으로 연결(Concatenate)한다.
  2. 공유된 컨볼루션 $\text{Conv}_{\theta_{C1}}$을 적용하여 상호작용 맵 $z_s$를 생성한다.
  3. 모든 $z_s$의 평균을 내어 쿼리 표현 $u'$를 업데이트한다.
  $$u' = \sigma(\text{Norm}(\text{Conv}_{\theta_{C3}}(\frac{1}{S} \sum_{s=1}^{S} z_s)))$$
  $$z_s = \text{Conv}_{\theta_{C1}}(u \parallel V^{feature}_s)$$
  또한, $\text{Conv}_{\theta_{C2}}$를 통해 서포트 표현 $V^{feature}_s$ 자체도 업데이트하여 계층적 특징을 추출한다.

### 학습 절차 및 데이터 생성
- **합성 데이터 생성**: Convolutional modeling을 통해 공간적으로 연속적인 21개의 CDP gather로 구성된 15,000개의 지진파 라인을 생성하였다.
- **데이터 증강 및 정규화**: Random white noise 추가 및 각 gather의 평균과 표준편차를 이용한 정규화를 수행하였다.
- **Identity Mapping 정규화**: 모델이 단순히 demultiple 변환만 배우는 것이 아니라 서포트 세트 $V$에 조건화되도록, 특정 확률로 레이블을 입력값 자체로 대체하여 "입력을 그대로 출력"하게 하는 정규화 기법을 적용하였다.
- **손실 함수**: $L_1$ 손실 함수를 사용하여 예측값 $Y^*$와 실제 레이블 $Y$ 사이의 차이를 최소화하였다.

## 📊 Results

### 실험 설정
- **비교 대상**: 표준 U-Net baseline.
- **지표**: PSNR(Peak Signal-to-Noise Ratio) 및 정성적 시각화 분석.
- **데이터**: 합성 데이터(Evaluation set 15%) 및 실제 현장 데이터(Field data, North Sea).

### 주요 결과
1. **정량적 성능 (합성 데이터)**:
   - 모든 CDP 위치에서 ContextSeisNet이 U-Net baseline보다 높은 PSNR을 기록하였다.
   - 서포트 세트의 크기($S$)가 증가할수록 전반적인 성능이 향상되는 경향을 보였으며, 프롬프트 간의 간격이 넓을수록 더 넓은 범위에서 일관된 성능을 유지하였다.
2. **정성적 성능 (합성 데이터)**:
   - U-Net과 달리 인접 CDP 간의 결과가 매우 일관적이며, 특히 $0.25\sim0.5$초 구간의 공간적 연속성이 크게 개선되었다.
   - Near-offset(근거리 오프셋) 영역에서의 다중 반사파 제거 성능이 월등히 향상되었다.
3. **현장 데이터 검증**:
   - 전통적인 Radon 방법과 U-Net baseline 모두에서 나타난 측면 불일치(특정 이벤트가 어떤 CDP에서는 제거되고 인접 CDP에서는 남는 현상)가 ContextSeisNet에서는 해결되었다.
   - **데이터 효율성**: 기존 U-Net이 일반화를 위해 약 100,000개의 gather가 필요했던 반면, ContextSeisNet은 단 10,500개의 gather(90% 적은 데이터)만으로도 대등하거나 더 우수한 현장 데이터 성능을 보여주었다.

## 🧠 Insights & Discussion

### 강점 및 해석
- **측면 일관성 확보**: ICL을 통해 인접 데이터의 정보를 추론에 활용함으로써, 지진파 데이터의 핵심 특성인 공간적 연속성을 보존하는 데 성공하였다.
- **사용자 제어 및 유연성**: 사용자가 Radon Transform 결과 등을 프롬프트로 제공함으로써, 딥러닝 모델의 출력을 도메인 전문가의 지식에 따라 가이드할 수 있는 가교 역할을 한다.
- **극도로 높은 데이터 효율성**: 추론 시점에 컨텍스트 정보를 활용하기 때문에, 학습 단계에서 방대한 양의 데이터를 외울 필요가 없어 학습 데이터 요구량이 획기적으로 줄어들었다.

### 한계 및 향후 과제
- **프롬프트 선택 전략**: 현재는 고정된 위치의 프롬프트를 사용하나, 실제 대규모 데이터셋에서는 어떤 CDP를 프롬프트로 선택할지에 대한 적응적 전략(Adaptive re-prompting)이 필요하다.
- **오류 전파 문제**: 이전 단계의 예측 결과를 다음 단계의 프롬프트로 사용하는 순차적 재프롬프팅(Sequential re-prompting) 시도 시, 오류가 누적되는 현상이 발견되었다.
- **계산 제약**: 학습 시 가변 크기의 서포트 세트를 사용하면 GPU 병렬화 효율이 떨어지므로, 현재는 고정 크기로 학습하고 추론 시에만 가변 크기를 적용하고 있다.

## 📌 TL;DR

본 논문은 지진파 다중 반사파 제거(Demultiple)에서 고질적인 문제인 **측면 불일치(Lateral Inconsistency)**와 **제어 불가능성**을 해결하기 위해 **In-Context Learning(ICL)**을 도입한 **ContextSeisNet**을 제안한다. 이 모델은 인접한 데이터 쌍을 프롬프트로 입력받아 추론 시점에 적응적으로 동작하며, 그 결과 **전통적 방법과 딥러닝의 장점을 결합**하여 공간적 일관성을 확보하고 **학습 데이터 요구량을 90% 감소**시키는 괄목할 만한 성과를 거두었다. 이 방법론은 향후 지진파 정렬(Alignment), 단층 검출(Fault detection) 등 다양한 지진파 처리 및 해석 작업으로 확장될 가능성이 매우 높다.