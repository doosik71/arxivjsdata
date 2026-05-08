# Efficient Adversarial Attacks for Visual Object Tracking

Siyuan Liang, Xingxing Wei, Siyuan Yao, and Xiaochun Cao (2020)

## 🧩 Problem to Solve

본 논문은 딥러닝, 특히 Siamese network 기반의 Visual Object Tracking (VOT) 모델들이 가진 취약성을 분석하고 이를 공격하는 효율적인 방법을 제안하는 것을 목표로 한다.

최근의 객체 추적기들은 DNN을 사용하여 높은 정확도를 달성하였으나, 이미지 분류(Image Classification)나 객체 검출(Object Detection) 분야와 달리 추적 모델의 강건성(Robustness)에 대한 연구는 거의 이루어지지 않았다. 특히 VOT는 기준 패치(Reference patch)와 후보 프레임 간의 유사도 측정(Similarity metric) 문제로 귀결되므로, 기존의 이미지 인식 작업 기반 공격 방법들을 그대로 적용하는 것은 한계가 있다. 따라서 본 연구는 VOT 작업에 특화된 Targeted attack과 Untargeted attack을 정의하고, 이를 실시간으로 수행할 수 있는 효율적인 공격 네트워크를 구축하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 Siamese 네트워크 기반 추적기의 취약점을 분석하여, 실시간으로 적대적 예제(Adversarial Examples)를 생성할 수 있는 **FAN (Fast Attack Network)**을 제안한 것이다.

핵심 아이디어는 추적기의 응답 맵(Response map)을 왜곡시키는 **Drift loss**와 특정 궤적으로 유도하기 위한 **Embedded feature loss**를 결합하여, 단일 GPU 환경에서도 매우 빠르게(약 10ms) 적대적 섭동(Perturbation)을 생성하는 엔드-투-엔드 네트워크를 설계한 점이다.

## 📎 Related Works

### 딥러닝 기반 객체 추적

현대적인 추적 시스템은 크게 Tracking-by-detection 프레임워크와 SiamFC 및 SiamRPN 기반의 Siamese 네트워크 구조로 나뉜다. 특히 Siamese 추적기들은 높은 국지화 정확도와 효율성을 보여주지만, 입력 데이터의 적대적 섭동에 매우 민감하다는 잠재적 취약점을 가지고 있다.

### 적대적 공격 방법론

기존의 적대적 공격은 크게 최적화 기반(Optimization-based) 방식과 생성 기반(Generator-based) 방식으로 구분된다. FGSM이나 PGD와 같은 최적화 방식은 그래디언트를 이용해 반복적으로 섭동을 생성하므로 연산 시간이 오래 걸린다. 반면 GAP나 UEA와 같은 생성 모델 기반 방식은 학습 후 추론 단계에서 빠르게 섭동을 생성할 수 있다. 저자들은 반복적 최적화 방식이 VOT와 같은 실시간 작업에는 속도 제한으로 인해 부적합하다고 지적하며, 생성 모델 기반의 접근 방식을 채택한다.

## 🛠️ Methodology

### 1. 문제 정의 (Problem Definition)

논문에서는 VOT에서의 공격을 다음과 같이 정의한다.

- **Targeted Attack**: 적대적 비디오 $\hat{V}$가 추적기로 하여금 객체를 미리 지정된 특정 궤적 $C_{spec}$을 따라 추적하도록 유도하는 것이다. 예측 중심점 $\hat{c}_i$와 목표 중심점 $c_{spec}^i$ 사이의 유클리드 거리가 임계값 $\epsilon$ (20 픽셀) 이내여야 한다.
- **Untargeted Attack**: 적대적 비디오 $\hat{V}$가 추적 결과 $B_{attack}$을 원래의 정답 궤적 $B_{gt}$로부터 완전히 벗어나게 하여, 예측 박스와 정답 박스의 IOU가 0이 되게 만드는 것이다.

### 2. Drift Loss Attack (Untargeted Attack용)

Siamese 추적기는 Fully-convolutional network를 통해 생성된 응답 맵 $S$의 최대 활성화 지점을 기반으로 객체 위치를 예측한다. 이를 이용해 응답 맵의 활성화 중심을 의도적으로 이동(Drift)시키는 전략을 사용한다.

- **Score Loss**: 응답 맵의 중심 영역(정답 영역)의 점수는 낮추고, 나머지 영역의 점수를 높여 활성화 중심을 이동시킨다.
$$L_{score}(G) = \min_{p \in S_{+1}} (l(y[p], s[p])) - \max_{p \in S_{-1}} (l(y[p], s[p]))$$
여기서 $l(y, s) = \log(1 + \exp(-ys))$이다.
- **Distance Loss**: 활성화 중심이 정답 중심에서 가능한 한 멀어지도록 유도한다.
$$L_{dist}(G) = \frac{\beta_1}{\delta + \|p_{+1}^{max} - p_{-1}^{max}\|^2} - \xi$$
- **최종 Drift Loss**: $$L_{drift} = L_{dist} + \beta_2 L_{score}$$

### 3. Embedded Feature Loss Attack (Targeted Attack용)

특정 궤적을 따라가게 하려면 응답 값 자체를 높여야 한다. 이를 위해 적대적 예제의 특징(Feature)이 타겟 이미지 $e$의 특징과 유사해지도록 강제한다.
$$L_{embed}(G) = \|\phi(q + G(q)) - \phi(e)\|^2$$
여기서 $q$는 입력 이미지, $G(q)$는 생성된 섭동, $\phi$는 특징 추출 함수이다. 타겟 특징 $e$를 위해 가우시안 노이즈를 사용하여 최적화를 진행함으로써 섭동의 가시성을 낮추었다.

### 4. FAN (Fast Attack Network) 구조 및 통합 학습

FAN은 CycleGAN의 구조를 참고한 생성기(Generator)와 PatchGAN 기반의 판별기(Discriminator)로 구성된다.

- **판별기 손실 함수**: 생성된 이미지가 실제 이미지인지 구분하도록 학습한다.
$$L_D(G, D, X) = \mathbb{E}_{x \sim p_{data}(x)} [(D(G(x) + x))^2] + \mathbb{E}_{x \sim p_{data}(x)} [(D(x) - 1)^2]$$
- **생성기 손실 함수**: 판별기를 속이고, 원본 이미지와의 시각적 유사성을 유지하며, 위에서 정의한 공격 손실들을 최소화한다.
$$L = L_G + \alpha_1 L_{sim} + \alpha_2 L_{embed} + \alpha_3 L_{drift}$$
여기서 $L_{sim} = \mathbb{E}[\|X - \hat{X}\|^2]$는 시각적 유사성을 위한 $L_2$ 거리 손실이다. 하이퍼파라미터 $\alpha_2, \alpha_3$를 조정함으로써 Targeted attack과 Untargeted attack을 선택적으로 혹은 동시에 수행할 수 있다.

## 📊 Results

### 실험 설정

- **데이터셋**: OTB2013, OTB2015, VOT2014, VOT2018.
- **모델**: White-box 공격 대상으로는 SiamFC(Alexnet 기반)를, Black-box 공격 대상으로는 SiamRPN, SiamRPN+CIR, SiamRPN++를 사용하였다.
- **지표**: Success score, Precision score, Success rate, Mean-Failures, Mean-SSIM.

### 주요 결과

1. **Untargeted Attack**: OTB 데이터셋에서 Success score, Precision score 등이 최소 72% 이상 하락하는 높은 공격 성공률을 보였다. 특히 VOT2018 데이터셋에서는 Mean-Failures가 413%나 증가하여 추적기가 객체를 매우 빈번하게 놓치게 만들었다.
2. **Targeted Attack**: 자동으로 생성된 가장 어려운 궤적(정답과 완전히 반대되는 방향)에 대해서도 추적기가 해당 궤적을 따라가게 만드는 데 성공하였다.
3. **실시간성 및 가시성**: 적대적 예제 생성 시간이 샘플당 약 10ms에 불과하여 실시간 공격이 가능하며, Mean-SSIM 수치가 1에 가까워(최대 7% 하락) 인간의 눈으로는 섭동을 거의 감지할 수 없음을 확인하였다.
4. **기존 방법론과의 비교**: modified FGSM 및 PGD와 비교했을 때, FAN은 공격 성공률(Drop rate) 면에서 압도적이며, 특히 PGD(샘플당 3.5s) 대비 속도가 비약적으로 빠르다.
5. **전이성(Transferability)**: SiamFC에서 생성한 섭동이 SiamRPN 계열에서도 작동함을 확인하였다. 다만, 구조적으로 공간적 오차를 보정하는 기능이 있는 SiamRPN++에서는 공격 효율이 상대적으로 낮게 나타났다.

## 🧠 Insights & Discussion

본 논문은 Siamese 추적기가 응답 맵의 최대 활성화 지점에 과도하게 의존한다는 점을 정확히 파고들었다. 특히 단순한 섭동 추가가 아니라, 생성 모델을 통해 특징 공간(Feature space)에서의 거리를 조절함으로써 타겟 궤적을 유도했다는 점이 인상적이다.

**강점**으로는 반복적 최적화의 느린 속도 문제를 GAN 구조로 해결하여 실시간 공격 가능성을 입증한 점과, Targeted/Untargeted 공격을 하나의 프레임워크 내에서 손실 함수 가중치 조절만으로 통합 구현한 점을 들 수 있다.

**한계 및 논의사항**으로는 Black-box 공격 시 모델의 아키텍처가 복잡해질수록(예: SiamRPN++) 전이성이 떨어진다는 점이 명시되었다. 이는 추적기의 구조적 개선이 적대적 공격에 대한 일종의 방어 기제로 작용할 수 있음을 시사한다. 또한, Targeted attack의 경우 궤적을 자동으로 생성하여 실험하였으나, 실제 환경에서 어떤 정교한 궤적을 설정했을 때 최적의 공격이 가능할지에 대한 추가 연구가 필요해 보인다.

## 📌 TL;DR

본 논문은 Siamese 네트워크 기반의 시각적 객체 추적기(VOT)를 공격하기 위한 실시간 생성 네트워크인 **FAN**을 제안한다. **Drift loss**를 통해 추적 대상의 위치를 이탈시키고, **Embedded feature loss**를 통해 추적기를 특정 궤적으로 유도한다. 실험 결과, 인간이 인지하기 어려운 수준의 미세한 섭동만으로 추적 성능을 70% 이상 저하시키거나 원하는 방향으로 유도할 수 있었으며, 생성 속도가 매우 빨라(10ms) 실제 환경에서의 위협 가능성을 보여주었다. 이 연구는 향후 딥러닝 기반 추적 모델의 강건성을 높이기 위한 방어 기법 연구에 중요한 기초 자료가 될 것이다.
