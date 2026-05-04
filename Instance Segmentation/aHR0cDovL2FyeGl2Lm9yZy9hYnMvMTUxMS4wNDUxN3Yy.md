# Reversible Recursive Instance-level Object Segmentation

Xiaodan Liang, Yunchao Wei, Xiaohui Shen, Zequn Jie, Jiashi Feng, Liang Lin, Shuicheng Yan (2015)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 문제는 **Instance-level Object Segmentation**이다. 이는 단순한 Semantic Segmentation을 넘어, 이미지 내의 동일 카테고리에 속하는 개별 객체(Instance)들을 구분하여 픽셀 단위로 마스킹하는 작업이다. 

이 문제가 어려운 이유는 객체들이 다양한 크기(scale)와 포즈(pose)를 가지며, 심한 가려짐(heavy occlusion)이나 불분명한 경계선 문제를 가지고 있기 때문이다. 기존의 많은 접근 방식은 Object Proposal(객체 제안 영역) 생성 단계와 이후의 세그멘테이션 단계를 분리하여 처리하는 파이프라인을 사용한다. 이로 인해 제안 영역의 품질이 최종 세그멘테이션 성능을 제한하는 병목 현상이 발생한다.

논문의 목표는 **Object Proposal의 정교화(Refinement)와 Instance-level Segmentation을 하나의 통합된 프레임워크 내에서 상호 보완적으로 학습**시켜, 제안 영역의 정확도를 높임과 동시에 정밀한 객체 마스크를 생성하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Reversible Recursive** 구조를 통해 제안 영역과 세그멘테이션 결과를 반복적으로 업데이트하는 것이다.

1. **Recursive Learning**: 세그멘테이션 네트워크와 제안 영역 정교화 네트워크를 재귀적으로 연결하여, 정교해진 제안 영역이 더 나은 세그멘테이션을 가능하게 하고, 다시 정교해진 세그멘테이션 결과가 제안 영역의 위치를 더 정확하게 보정하는 상호 이득 구조를 설계하였다.
2. **Reversible Proposal Refinement**: 모든 제안 영역에 대해 동일한 횟수의 반복을 수행하는 대신, 각 제안 영역의 상태에 따라 최적의 반복 횟수를 적응적으로 결정하는 Reversible Gate를 도입하였다.
3. **Instance-aware Denoising Autoencoder**: 하나의 제안 영역 내에 여러 객체가 겹쳐 있을 때, 가장 지배적인(dominant) 객체만을 정확히 추출하기 위해 전역 정보(global information)를 활용하는 Denoising Autoencoder를 세그멘테이션 네트워크에 통합하였다.

## 📎 Related Works

**Object Detection** 분야에서는 Selective Search나 RPN과 같은 Proposal 생성 기법과 이를 이용한 분류 및 Bounding Box 회귀(Regression) 방식이 주로 사용된다. 그러나 이러한 방식은 제안 영역 생성과 탐지를 별개의 단계로 취급하여 최적의 결과를 얻기 어렵다는 한계가 있다.

**Instance-level Object Segmentation** 연구들(SDS, HC 등)은 대부분 고정된 Object Proposal을 전제로 하는 single-pass feed-forward 구조를 가진다. 반면 R2-IOS는 제안 영역을 반복적으로 정교화함으로써 기존의 고정적 접근 방식의 한계를 극복한다. 또한, Proposal-free 방식(PFN 등)은 객체 수 예측의 정확도에 의존하므로 작은 객체를 탐지하는 데 취약하지만, R2-IOS는 정교화된 Proposal을 기반으로 하여 작은 객체에 대해서도 더 높은 커버리지를 가진다.

## 🛠️ Methodology

### 전체 시스템 구조
R2-IOS는 VGG-16 모델을 기반으로 하며, 크게 두 개의 서브 네트워크로 구성된다: **Instance-level Segmentation Sub-network**와 **Reversible Proposal Refinement Sub-network**. 두 네트워크는 재귀적으로 연결되어 $T$번의 반복 학습을 수행한다.

### 1. Instance-level Segmentation Sub-network
이 네트워크는 제안 영역 내의 지배적인 객체에 대한 foreground mask를 생성한다.
- **구조**: VGG-16에서 마지막 두 개의 Max pooling 레이어를 제거하여 국부적 디테일을 유지하고, Fully-connected 레이어를 Fully-convolutional 레이어로 대체하여 특징 맵을 생성한다.
- **ROI Pooling**: 다양한 크기의 Proposal을 $40 \times 40$ 크기의 고정된 특징 맵으로 변환한다.
- **Instance-aware Denoising Autoencoder**: 
    - $1 \times 1$ Convolution을 통해 얻은 신뢰도 맵 $C$를 벡터 $\tilde{C}$(차원 $40 \times 40 \times 2$)로 변환한다.
    - **Encoder**: $\tilde{C}$를 비선형 연산자 $\Phi(\cdot)$를 통해 저차원 은닉 표현 $h = \Phi(\tilde{C})$(512차원)로 매핑한다.
    - **Decoder**: $h$를 다시 $\Phi'(\cdot)$를 통해 재구성된 벡터 $v = \Phi'(h)$(3200차원)로 매핑하고, 이를 다시 $40 \times 40$ 맵으로 변형한다.
    - 이 과정은 제안 영역 내의 여러 객체 중 가장 지배적인 객체의 마스크만을 남기고 나머지를 노이즈로 간주하여 제거하는 역할을 한다.

### 2. Reversible Proposal Refinement Sub-network
이 네트워크는 객체의 카테고리 신뢰도와 Bounding Box 오프셋을 예측하여 Proposal 위치를 수정한다.
- **Segmentation-aware Features**: 단순히 이미지 특징만 사용하는 것이 아니라, 세그멘테이션 네트워크에서 생성된 마스크 $v$와 정교화 네트워크의 마지막 FC 레이어 특징을 결합(concatenate)하여 사용한다. 이를 통해 픽셀 단위의 경계 정보가 Proposal의 위치 보정에 기여하게 한다.
- **Reversible Gate**: 
    - 모든 반복($t=1, \dots, T$)을 수행한 후, 카테고리 신뢰도가 가장 높았던 시점 $t'$를 최적 반복 횟수로 결정한다.
    - 테스트 시에는 $t'$-번째 결과물을 최종 출력으로 사용하며, 학습 시에는 $t'$ 이전까지의 손실(loss)만 사용하여 파라미터를 업데이트하고 그 이후의 손실은 버린다.

### 3. Recursive Learning 및 손실 함수
초기 Proposal $l_0$부터 시작하여 $t$-번째 반복에서 예측된 오프셋 $o_{t,k}$를 이용해 $l_t$를 갱신한다. 각 반복 단계에서의 총 손실 함수 $J_t$는 다음과 같다.

$$J_t = J_{cls}(p_t, g) + \mathbb{1}[g \ge 1]J_{loc}(o_t, g, \tilde{o}_t) + \mathbb{1}[g \ge 1]J_{seg}(v_t, \tilde{v}_t)$$

여기서:
- $J_{cls}$: 클래스 분류를 위한 Log loss이다.
- $J_{loc}$: Bounding Box 위치 보정을 위한 Smooth $L_1$ loss이다.
- $J_{seg}$: 픽셀 단위 세그멘테이션을 위한 Cross-entropy loss이다.
- $\mathbb{1}[g \ge 1]$은 해당 Proposal이 배경이 아닌 객체를 포함하고 있을 때만 위치 및 세그멘테이션 손실을 계산함을 의미한다.
- 최종 전역 손실은 $J = \sum_{t \le t'} J_t$로 계산된다.

## 📊 Results

### 실험 설정
- **데이터셋**: PASCAL VOC 2012 validation set 및 SBD 데이터셋을 사용하였다.
- **평가 지표**: $AP_r$ (Average Precision)을 사용하였으며, IoU 임계값을 0.5, 0.6, 0.7로 설정하여 측정하였다.
- **비교 대상**: SDS, HC, PFN 및 Chen et al.의 방법론.

### 정량적 결과
- **0.5 IoU 기준**: R2-IOS는 **66.7%**의 $AP_r$을 달성하여, PFN(58.7%)과 Chen et al.(46.3%) 및 SDS(43.8%)를 크게 상회하였다.
- **높은 IoU 기준**: 0.6 및 0.7 IoU 환경에서도 R2-IOS는 타 모델 대비 월등한 성능을 보였다. 특히 0.7 IoU에서 HC 대비 약 7.1%의 성능 향상을 보였다.

### Ablation Study 결과
- **Recursive Learning**: 반복 횟수가 1회에서 4회로 증가함에 따라 성능이 지속적으로 향상되었으며, 학습 단계에서 recursive training을 수행하는 것이 테스트 시에만 적용하는 것보다 3.3% 더 높은 성능을 보였다.
- **Reversible Gate**: 적응적 반복 횟수 결정 기능을 추가했을 때, 고정 4회 반복 대비 1.5%의 성능 향상이 있었다.
- **Instance-aware Autoencoder**: 이 모듈을 제거했을 때 성능이 12.5%나 하락하여, 전역 정보를 이용한 지배적 객체 추출이 매우 중요함을 입증하였다.
- **Segmentation-aware Feature**: 세그멘테이션 정보를 정교화 네트워크에 피드백했을 때 성능이 향상되어, 두 네트워크 간의 상호 보완적 관계가 확인되었다.

## 🧠 Insights & Discussion

본 논문은 Object Proposal과 Instance Segmentation이라는 두 가지 작업을 분리하지 않고, **재귀적 구조(Recursive structure)**를 통해 상호 유기적으로 연결함으로써 성능을 극대화하였다. 특히, 단순한 반복이 아니라 **Reversible Gate**를 통해 각 객체마다 필요한 최적의 정교화 횟수를 다르게 적용한 점이 매우 효율적이다.

또한, Instance-level segmentation의 고질적인 문제인 **'겹쳐진 객체들 사이의 구분'** 문제를 해결하기 위해 Denoising Autoencoder를 도입하여 전역 문맥(global context)을 파악하고 지배적 객체를 추출해낸 점은 기술적으로 매우 유용한 접근이다.

**한계점 및 논의사항**:
- 본 연구는 최대 반복 횟수를 $T=4$로 설정하였는데, 더 많은 반복이 성능 향상을 가져오는지에 대한 분석은 부족하다.
- Proposal 생성 단계에서 Selective Search라는 전통적인 방식을 사용하였는데, 이를 최신 RPN이나 다른 딥러닝 기반 Proposal 생성기로 대체했을 때의 시너지 효과에 대한 논의가 필요하다.
- 추론 속도가 이미지당 약 1초(Proposal 생성 시간 제외)로 측정되었으나, 실시간 서비스에 적용하기에는 여전히 무거운 구조일 수 있다.

## 📌 TL;DR

R2-IOS는 **제안 영역 정교화**와 **인스턴스 세그멘테이션**을 재귀적으로 결합한 프레임워크이다. **Reversible Gate**를 통해 객체별 최적 반복 횟수를 적응적으로 결정하고, **Instance-aware Denoising Autoencoder**를 통해 겹쳐진 객체 중 지배적 객체를 정확히 분리해낸다. PASCAL VOC 2012 벤치마크에서 $AP_r$ 66.7%를 기록하며 기존 SOTA 모델들을 압도하였으며, 이는 통합된 재귀적 학습의 중요성을 시사한다. 향후 LSTM 등을 활용한 공간적 문맥 의존성 연구로 확장될 가능성이 크다.