# Efficient Medical Image Segmentation Based on Knowledge Distillation

Dian Qin, Jia-Jun Bu, Zhe Liu, Xin Shen, Sheng Zhou, Jing-Jun Gu, Zhi-Hua Wang, Lei Wu, Hui-Fen Dai (2021)

## 🧩 Problem to Solve

의료 영상 분할(Medical Image Segmentation) 분야에서 최근 합성곱 신경망(CNN)의 발전으로 정밀한 예측이 가능해졌으나, 대부분의 고성능 모델들은 거대한 계산 복잡도와 방대한 저장 공간을 요구한다. 이러한 특성은 실제 의료 현장의 실시간 배포 및 적용에 있어 심각한 제약이 된다. 

이를 해결하기 위해 모델 경량화 연구가 진행되고 있으나, 일반적으로 모델을 단순화하면 분할 성능이 저하되는 딜레마가 발생한다. 특히 의료 영상은 장기의 외형이 다양하고 크기가 불규칙하며, 경계가 모호한 특성이 있어 일반적인 이미지 분할보다 훨씬 까다롭다. 따라서 본 논문의 목표는 지식 증류(Knowledge Distillation) 기법을 통해 고성능의 교사 네트워크(Teacher Network)가 가진 지식을 경량화된 학생 네트워크(Student Network)로 효율적으로 전달하여, 추론 효율성을 유지하면서도 분할 성능을 획기적으로 끌어올리는 시스템을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 의료 영상의 특성을 고려하여 단순한 결과값(Logits)뿐만 아니라 중간 특징 맵(Intermediate Feature Maps)의 구조적 정보를 함께 전달하는 통합 지식 증류 아키텍처를 설계한 것이다. 

특히, 의료 영상에서 각 조직 영역 간의 시맨틱 차이가 뚜렷하다는 점에 착안하여, 영역 간의 관계 정보를 인코딩하여 전달하는 **Region Affinity Distillation (RAD)** 모듈을 제안하였다. 이는 의료 영상의 고질적인 문제인 모호한 경계(Ambiguous Boundary) 문제를 회피하고, 각 시맨틱 영역의 내부 정보를 직접 전이함으로써 학생 네트워크가 교사 네트워크의 분할 능력을 간접적으로 학습하게 만든다.

## 📎 Related Works

기존의 의료 영상 분할 연구들은 주로 UNet 계열의 아키텍처에 Dense Connection을 추가하거나, Attention 메커니즘을 도입하고, 2D 커널을 3D로 확장하는 방향으로 발전해 왔다. 그러나 이러한 방식들은 필연적으로 계산 비용을 증가시킨다.

지식 증류(Knowledge Distillation) 관점에서는 주로 최종 출력층의 Logits를 전이하는 방식이 사용되었으며, 최근에는 중간 특징 맵을 활용하려는 시도(Attention Transfer 등)가 있었다. 하지만 기존의 지식 증류 방식들은 의료 영상 특유의 복잡한 시맨틱 구조를 충분히 고려하지 않았으며, 대개 고정된 네트워크 구조를 요구하거나 단일 모달리티의 일반적인 의료 영상 분할 시스템을 체계적으로 구축한 사례가 부족했다는 한계가 있다.

## 🛠️ Methodology

본 논문이 제안하는 지식 증류 파이프라인은 교사 네트워크와 학생 네트워크가 동일한 이미지를 입력받아 각각 예측을 수행하며, 그 사이에서 세 가지 핵심 증류 모듈(PMD, IMD, RAD)이 작동하는 구조이다.

### 1. Prediction Maps Distillation (PMD)
교사 네트워크의 최종 출력 결과(Segmentation Map)를 학생 네트워크가 직접 모방하도록 유도한다. 픽셀 수준의 분류 문제로 간주하여, 두 네트워크 간의 확률 분포 차이를 Kullback-Leibler (KL) Divergence로 계산한다.

$$L_{PM} = \frac{1}{N} \sum_{i \in N} KL(p^s_i || p^t_i)$$

여기서 $N$은 전체 픽셀 수이며, $p^s_i$와 $p^t_i$는 각각 학생과 교사 네트워크가 예측한 $i$번째 픽셀의 확률값이다.

### 2. Importance Maps Distillation (IMD)
네트워크의 중간 특징 맵에서 중요한 뉴런의 활성화 패턴을 전이한다. 교사와 학생의 특징 맵 크기가 다를 수 있으므로, 먼저 리스케일링(Rescaling) 과정을 통해 공간적 크기를 맞춘다. 이후 채널 차원을 따라 절대값의 합을 구하여 중요도 맵(Importance Map) $M$을 생성한다.

$$\phi(\varepsilon) = \sum_{i=1}^{C} |\varepsilon_i|$$

두 네트워크의 중요도 맵 간의 차이를 $L_1$ 및 $L_2$ 정규화를 통해 계산하여 손실 함수 $L_{IM}$을 정의한다.

### 3. Region Affinity Distillation (RAD)
본 논문의 핵심 모듈로, 서로 다른 시맨틱 영역 간의 관계(Affinity) 정보를 전이한다. 
- **영역 정보 벡터 추출**: 정답 마스크(Ground Truth Mask)를 이용하여 특정 클래스 $i$에 해당하는 픽셀들의 특징 값들을 평균 내어 영역 정보 벡터 $R_i$를 생성한다.
  $$R_i = \frac{1}{N_i} \sum_{j=1}^{w \times h} \varepsilon_j \cdot m_{ij}$$
- **영역 대비 값 계산**: 서로 다른 클래스 영역 벡터 간의 코사인 유사도를 측정하여 영역 대비 값 $V_{rc}$를 구한다.
  $$V_{rc} = \frac{1}{n} \sum_{(i,j)} \frac{R_i^T R_j}{||R_i||_2 ||R_j||_2}$$
- **손실 함수**: 학생 네트워크의 $V_{rc}^s$가 교사의 $V_{rc}^t$를 모방하도록 $L_{RA}$를 정의한다.
  $$L_{RA} = \sum_{(i,j) \in P} ||V_{rc}^s - V_{rc}^t||_p$$

### 4. 학습 절차 및 전체 손실 함수
전체 시스템은 end-to-end 방식으로 학습되며, 최종 손실 함수는 일반적인 분할 손실($L_{seg}$)과 세 가지 증류 손실의 가중 합으로 구성된다.

$$L_{total} = L_{seg} + \alpha L_{PM} + \beta_1 L_{IM} + \beta_2 L_{RA}$$

가중치는 $\alpha=0.1, \beta_1=0.9, \beta_2=0.9$로 설정되었다. 학습이 완료되면 교사 네트워크와 증류 모듈은 모두 제거되고, 최적화된 경량 학생 네트워크만 추론에 사용된다.

## 📊 Results

### 실험 설정
- **데이터셋**: LiTS(간 및 간 종양 CT)와 KiTS19(신장 및 신장 종양 CT) 공공 데이터셋 사용.
- **네트워크 구성**: 
    - 교사(Teacher): RA-UNet, PSPNet, UNet++
    - 학생(Student): ENet, MobileNetV2, ResNet-18
- **평가 지표**: Dice Coefficient (주 지표), Volume Overlap Error (VOE), Relative Volume Difference (RVD).

### 주요 결과
1. **성능 향상**: 지식 증류를 통해 학생 네트워크의 성능이 비약적으로 상승했다. 특히 신장 종양 분할 실험에서 MobileNetV2의 Dice 계수가 $0.516$에서 $0.684$로 최대 $32.6\%$ 향상되었다.
2. **효율성**: 학생 네트워크(예: ENet)는 교사 네트워크(예: RA-UNet)보다 파라미터 수가 약 21배 적음에도 불구하고, 성능 격차를 3.75배(0.229 $\rightarrow$ 0.061)로 좁혔다.
3. **SOTA 비교**: RA-UNet으로부터 지식 증류된 ENet은 파라미터 수가 매우 적음에도 불구하고, 신장 종양 분할 작업에서 RA-UNet을 제외한 거의 모든 모델(PSPNet, DeeplabV3+ 등)보다 우수한 성능을 보였다.
4. **모듈 효과**: Ablation Study 결과, PMD, IMD, RAD 세 모듈을 모두 사용했을 때 가장 높은 성능을 보였으며, 특히 IMD와 RAD가 성능 향상에 핵심적인 역할을 함이 입증되었다.

## 🧠 Insights & Discussion

본 연구는 의료 영상 분할을 위한 체계적인 지식 증류 아키텍처를 제안함으로써, 경량 모델의 한계를 극복할 수 있음을 보여주었다. 특히 RAD 모듈은 픽셀 단위의 경계 예측에 집착하는 대신 영역 간의 관계 정보를 학습하게 함으로써 의료 영상 특유의 모호한 경계 문제를 효과적으로 해결했다.

흥미로운 점은 일부 실험에서 지식 증류된 학생 네트워크가 원래의 교사 네트워크보다 더 높은 성능을 기록했다는 것이다(예: MobileNetV2 $\rightarrow$ PSPNet/UNet++). 이는 증류 과정이 학생 네트워크가 시맨틱 정보를 더 잘 이해하도록 가이드하는 일종의 정규화(Regularization) 효과를 제공했기 때문으로 추측된다.

다만, 본 논문은 2D 슬라이스 기반의 분석에 집중하고 있어 3D 시나리오로 확장할 경우 계산 복잡도와 메모리 사용량이 급증하는 문제가 발생할 수 있다. 또한, 3D 환경에서는 객체 영역과 배경의 비율이 훨씬 낮아져 유의미한 정보를 전이하는 것이 더 어려울 것이라는 한계점이 존재한다.

## 📌 TL;DR

본 논문은 거대 모델의 성능과 경량 모델의 효율성을 동시에 잡기 위해, **Prediction Map(PMD), Importance Map(IMD), 그리고 영역 간 관계를 다루는 Region Affinity(RAD)** 세 가지 모듈을 결합한 지식 증류 아키텍처를 제안한다. 실험 결과, 학생 네트워크의 파라미터 수는 획기적으로 줄이면서도 분할 성능은 최대 $32.6\%$ 향상시켜, 실제 의료 현장에서의 실시간 배포 가능성을 높였다. 이는 향후 의료 영상 처리의 실용적인 경량화 모델 연구에 중요한 기반이 될 것으로 보인다.