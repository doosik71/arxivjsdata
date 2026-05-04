# SiamMan: Siamese Motion-aware Network for Visual Tracking

Wenzhang Zhou, Longyin Wen, Libo Zhang, Dawei Du, Tiejian Luo, Yanjun Wu (2020)

## 🧩 Problem to Solve

본 논문은 비주얼 객체 추적(Visual Object Tracking)에서 발생하는 급격한 움직임(Abrupt Motion), 폐색(Occlusion), 그리고 조명 변화와 같은 도전적인 상황을 해결하고자 한다. 특히, 기존의 Siamese-RPN 계열 알고리즘들이 사전에 정의된 앵커 박스(Anchor boxes)에 의존하여 바운딩 박스를 회귀(Regression)시키는 방식에 주목한다. 이러한 방식은 타겟의 다양한 움직임 패턴과 스케일 변화에 유연하게 대응하지 못하며, 특히 타겟이 빠르게 움직이거나 일시적으로 가려지는 상황에서 추적 실패로 이어지는 한계가 있다. 따라서 본 연구의 목표는 타겟의 움직임을 인식할 수 있는 메커니즘을 도입하여, 다양한 모션 패턴에서도 강건하게 타겟을 추적할 수 있는 Siamese Motion-aware Network(SiamMan)를 설계하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 타겟의 정밀한 바운딩 박스 회귀를 돕기 위해 '거친 위치 추정(Coarse Localization)' 단계를 추가하는 것이다. 주요 기여 사항은 다음과 같다.

1.  **삼중 병렬 브랜치 구조**: 기존의 분류(Classification) 및 회귀(Regression) 브랜치 외에, 타겟의 중심 위치를 대략적으로 찾아내는 Localization 브랜치를 추가하여 전체 시스템을 구성하였다.
2.  **Global Context Module 도입**: Localization 브랜치 내에 Global Context Module을 통합하여 장거리 의존성(Long-range dependency)을 캡처함으로써, 타겟의 변위(Displacement)가 큰 상황에서도 강건한 추적이 가능하게 하였다.
3.  **Multi-scale Learnable Attention Module 설계**: 서로 다른 레이어에서 추출된 다중 스케일 특징들을 효과적으로 활용하기 위해, 학습 가능한 가중치를 가진 어텐션 모듈을 설계하여 각 브랜치가 변별력 있는 특징을 추출하도록 유도하였다.

## 📎 Related Works

비주얼 추적 연구는 크게 상관 필터(Correlation Filter, CF) 기반 방식과 딥 컨볼루션 네트워크(CNN) 기반 방식으로 나뉜다. CF 기반 방식은 계산 효율성이 높지만, 최근의 딥러닝 기반 방식들이 더 높은 정확도를 보여주고 있다. 특히 Siamese 네트워크를 이용한 방식들은 추적 문제를 원샷 검출(One-shot detection) 문제로 정의하여 효율성과 정확도를 동시에 잡으려 노력해 왔다.

SiamRPN 및 SiamRPN++와 같은 최신 모델들은 RPN(Region Proposal Network) 구조를 통해 타겟을 검출하지만, 앞서 언급한 바와 같이 수동으로 설계된 앵커 박스에 의존한다. 이는 타겟의 급격한 모션이나 스케일 변화가 발생했을 때 앵커 박스가 타겟을 제대로 커버하지 못하는 문제를 야기한다. 본 논문은 이러한 앵커 기반 방식의 한계를 극복하기 위해 Localization 브랜치를 통한 보조적인 위치 추정 방식을 제안하며 기존 연구와 차별점을 둔다.

## 🛠️ Methodology

### 전체 시스템 구조
SiamMan은 Siamese 특징 추출 서브네트워크(Backbone)와 그 뒤에 병렬로 연결된 세 개의 브랜치(Classification, Regression, Localization)로 구성된다. Backbone으로는 ResNet-50을 사용하며, 템플릿(Template) 모듈과 검출(Detection) 모듈이 파라미터를 공유하는 구조이다.

### 주요 구성 요소 및 역할

1.  **Classification Branch**: 타겟의 전경(Foreground)과 배경(Background)을 구분한다. 템플릿 특징 $\phi_{cls}^m(\alpha)$와 검출 특징 $\phi_{cls}^m(\beta)$ 간의 depth-wise convolution을 통해 상관 특징 맵(Correlation feature map)을 생성한다.
    $$F_{cls}^{w \times h \times 2k}(m) = \phi_{cls}^m(\alpha) * \phi_{cls}^m(\beta)$$
    이후 Multi-scale Attention 모듈을 통해 최종 예측값 $O_{cls}$를 산출한다.

2.  **Regression Branch**: 앵커 박스를 기반으로 타겟의 정밀한 바운딩 박스를 회귀한다. 작동 방식은 분류 브랜치와 유사하며, 최종적으로 각 앵커와 정답 박스 간의 정규화된 거리 값을 출력한다.
    $$O_{reg}^{w \times h \times 4k} = \sum_{m=1}^{L} \gamma_{reg}^m \cdot F_{reg}^{w \times h \times 4k}(m)$$

3.  **Localization Branch**: 타겟의 중심 위치를 거칠게 추정하여 회귀 브랜치를 보조한다. 특징 맵 간의 요소별 곱셈(Element-wise multiplication)을 사용하며, Global Context Module과 Atrous Spatial Pyramid Module을 통해 광범위한 문맥 정보를 통합한다.
    $$F_{loc}^{w \times h \times 2}(m) = E[\phi_{loc}^m(\alpha)] \odot \phi_{loc}^m(\beta)$$
    여기서 $E[\cdot]$는 크기 조정(Resize) 연산이며, $\odot$는 요소별 곱셈이다.

4.  **Multi-scale Learnable Attention Module**: 각 브랜치는 여러 레이어($m=1, \dots, L$)의 특징 맵을 사용한다. 각 스케일의 중요도를 나타내는 학습 가능한 가중치 $\gamma$를 통해 가중 합(Weighted sum)을 구함으로써 최적의 특징 조합을 생성한다.

### 손실 함수 (Loss Function)
전체 손실 함수는 세 브랜치의 손실 합으로 정의된다.
$$L = \lambda_{cls}L_{cls}(u, u^*) + \lambda_{reg}L_{reg}(p, p^*) + \lambda_{loc}L_{loc}(c, c^*)$$
- **$L_{cls}, L_{loc}$**: 타겟 유무 및 중심 위치 예측을 위해 Cross-entropy loss를 사용한다. 특히 $L_{loc}$의 정답 레이블 $c^*$는 Gaussian 커널을 이용하여 생성한다.
- **$L_{reg}$**: 바운딩 박스 좌표의 정밀한 회귀를 위해 $L1$ loss를 사용한다.

### 추론 절차 (Inference)
최종 예측 점수 $\Theta$는 다음과 같은 가중치 조합으로 결정된다.
$$\Theta^{w \times h \times k} = \omega_2 \cdot \rho \cdot (\omega_1 \cdot u + (1 - \omega_1) \cdot c) + (1 - \omega_2) \cdot \xi$$
여기서 $u$는 분류 결과, $c$는 위치 추정 결과, $\xi$는 경계 외부 이상치를 억제하기 위한 코사인 윈도우(Cosine window), $\rho$는 급격한 스케일 변화를 억제하는 페널티 항이다.

## 📊 Results

### 실험 설정
- **데이터셋**: VOT2016, VOT2018, OTB100, UAV123, LTB35 총 5개의 벤치마크를 사용하였다.
- **평가 지표**: EAO(Expected Average Overlap), Accuracy(A), Robustness(R), Success score, Precision score, F-score 등을 사용하여 성능을 측정하였다.
- **구현 환경**: PySOT 플랫폼 기반, NVIDIA RTX 2080 GPU 사용, 평균 추적 속도는 45fps이다.

### 정량적 결과
- **VOT2016 & VOT2018**: VOT2016에서 EAO 0.513, VOT2018에서 EAO 0.462를 기록하며 SOTA(State-of-the-art)를 달성하였다. 특히 SiamRPN++ 대비 EAO가 크게 향상되어 Localization 브랜치의 효과를 입증하였다.
- **OTB100 & UAV123**: OTB100에서 Success score 0.705, Precision score 0.919로 최상위 성능을 보였으며, UAV123에서는 DiMP-50과 대등한 성능을 기록하였다.
- **LTB35 (Long-term)**: 별도의 재검출(Re-detection) 모듈 없이도 F-score 64.1%를 달성하며 장기 추적 성능이 우수함을 보였다.

### 소거 연구 (Ablation Study)
Localization 브랜치, Global Context Module, Multi-scale Attention 각각의 제거 시 EAO가 유의미하게 하락함을 확인하였다. 특히 Localization 브랜치가 없을 때 성능 저하가 가장 컸으며, 이는 제안 방법의 핵심 기여분이 타겟의 거친 위치 추정에 있음을 시사한다.

## 🧠 Insights & Discussion

본 논문은 기존 Siamese 네트워크가 가진 '앵커 박스 의존성'이라는 근본적인 한계를 Localization 브랜치라는 보조 경로를 통해 효과적으로 완화하였다. 특히 Fast Motion, Out-of-view, Low Resolution과 같은 어려운 속성(Attribute)에서 타 모델 대비 월등한 성능을 보인 점은, 단순한 정밀 회귀보다 '먼저 대략적인 위치를 찾는 것'이 추적 실패율을 낮추는 데 결정적임을 보여준다.

다만, Accuracy 측면에서는 SiamMask와 같은 세그멘테이션 기반 방식보다 다소 낮은 수치를 보이는데, 이는 SiamMask가 마스크를 통해 회전된 바운딩 박스를 정밀하게 추정하기 때문으로 해석된다. 결과적으로 SiamMan은 정밀도(Accuracy)보다는 강건성(Robustness)과 실패율 감소에 더 최적화된 구조라고 볼 수 있다.

## 📌 TL;DR

SiamMan은 기존 Siamese-RPN의 고정된 앵커 박스 문제를 해결하기 위해 **Classification, Regression, Localization의 세 가지 병렬 브랜치**를 제안한 모델이다. 특히 Localization 브랜치에 **Global Context Module**을 도입하여 급격한 움직임과 큰 변위에도 강건하게 대응하며, **학습 가능한 Multi-scale Attention**을 통해 최적의 특징을 조합한다. 이를 통해 VOT2016, VOT2018 등 주요 벤치마크에서 SOTA를 달성하였으며, 특히 빠른 움직임이 포함된 시나리오에서 추적 실패율을 획기적으로 낮추었다.