# Adaptive Instance Distillation for Object Detection in Autonomous Driving

Qizhen Lan and Qing Tian (2023)

## 🧩 Problem to Solve

본 논문은 자율 주행 환경과 같이 실시간성이 중요한 시나리오에서 객체 탐지(Object Detection) 모델의 효율성을 높이기 위한 지식 증류(Knowledge Distillation, KD) 방법론을 다룬다. 일반적으로 지식 증류는 거대한 교사 모델(Teacher Model)의 지식을 경량화된 학생 모델(Student Model)에게 전달하여, 효율적이면서도 높은 성능을 내는 모델을 만드는 기법이다.

기존의 객체 탐지 지식 증류 방법들은 주로 어떤 형태의 지식(특성 맵, 예측값, 어텐션 맵 등)을 전달할 것인지에 집중했으며, 모든 인스턴스(Instance)를 동일한 비중으로 처리하는 경향이 있었다. 그러나 저자들은 교사 모델조차 모든 인스턴스를 동일하게 잘 학습하지 못한다는 점에 주목한다. 즉, 교사 모델이 확신하지 못하는 '불확실한' 인스턴스의 지식까지 학생 모델이 맹목적으로 학습할 경우, 오히려 성능 저하가 발생할 수 있다는 것이 본 연구가 해결하고자 하는 핵심 문제이다. 따라서 논문의 목표는 교사 모델의 예측 신뢰도에 따라 인스턴스별로 증류 가중치를 동적으로 조절하는 Adaptive Instance Distillation (AID) 방법을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **교사 모델의 예측 오차를 기반으로 학생 모델이 학습할 지식의 가중치를 결정**하는 것이다. 구체적인 기여 사항은 다음과 같다.

1.  **인스턴스 기반 선택적 학습**: 모든 데이터를 동일하게 학습하는 대신, 교사 모델이 정확하게 예측한 인스턴스의 지식은 적극적으로 수용하고, 교사 모델이 어려워하는 인스턴스에 대해서는 학생 모델이 스스로 학습할 수 있도록 가중치를 낮추는 방식을 제안한다.
2.  **Adaptive Instance Distillation (AID) 설계**: 교사 모델의 Task Loss를 이용하여 증류 가중치를 지수 함수적으로 조절하는 메커니즘을 설계하였다.
3.  **FPN 계층별 적용**: Feature Pyramid Networks (FPN)의 각 출력 레이어에 AID를 적용하여, 객체의 크기(Scale)에 따른 신뢰도를 반영한 scale-wise 선택적 지식 증류를 가능하게 하였다.
4.  **범용적 적용 가능성**: 단일 단계(Single-stage) 및 이 단계(Two-stage) 검출기 모두에 적용 가능하며, 동일 구조의 모델 간 지식을 전달하는 Self-distillation으로도 확장하여 성능을 향상시켰다.

## 📎 Related Works

본 논문에서는 객체 탐지, 적응형 인스턴스 가중치 부여, 지식 증류 세 가지 분야의 관련 연구를 검토한다.

*   **Object Detection**: Faster R-CNN과 같은 Two-stage detector와 GFL, YOLO, SSD와 같은 One-stage detector로 구분하며, 다양한 크기의 객체 탐지를 위해 FPN이 널리 사용되고 있음을 언급한다.
*   **Adaptive Instance Weighting**: Focal Loss나 GHM-C와 같이 어려운 샘플에 더 큰 가중치를 주는 Hard Example Mining 기법들이 존재한다. 그러나 Zhang et al. [10]은 이러한 Hard-mining 방식이 지식 증류에는 적합하지 않으며, 오히려 쉬운 샘플에 집중하는 것이 더 효과적일 수 있음을 시사하였다.
*   **Knowledge Distillation**: 기존 연구들은 주로 feature-based, response-based, relation-based 지식을 전달하는 방식에 집중했다. 특히 객체 탐지 분야의 KD는 분류 문제보다 복잡하며, 배경과 전경의 불균형 문제 등으로 인해 연구가 상대적으로 덜 진행되었다. 기존의 인스턴스 기반 가중치 연구(Zhang et al. [10])는 학생 모델의 보조 브랜치(auxiliary branch)에서 발생하는 특징 분산을 사용했으나, 본 논문은 이를 교사 모델의 예측 기반으로 대체하여 더 직접적인 신뢰도를 측정하고자 한다.

## 🛠️ Methodology

### 전체 시스템 구조
AID는 기존의 지식 증류 파이프라인에 통합될 수 있는 플러그인 형태의 가중치 조절 메커니즘이다. 교사 모델의 예측 결과와 정답(Ground Truth) 사이의 거리를 측정하여, 학생 모델의 증류 손실 함수에 곱해질 가중치를 생성한다.

### 손실 함수 및 학습 절차
학생 모델의 전체 학습 손실 함수 $L^S_i$는 다음과 같이 정의된다.

$$L^S_i = L^S_{task,i} + \lambda L^{S,T}_{AID,i}$$

여기서 $L^S_{task,i}$는 학생 모델이 원래 수행해야 하는 객체 탐지 작업 손실이며, $\lambda$는 task loss와 AID loss 사이의 균형을 맞추는 가중치 계수이다. 

핵심이 되는 AID 손실 함수 $L^{S,T}_{AID,i}$는 다음과 같다.

$$L^{S,T}_{AID,i} = \exp(-\alpha D^T_i) L^{S,T}_{distill,i}$$

이 식에서 각 변수의 의미는 다음과 같다.
*   $L^{S,T}_{distill,i}$: 교사 모델과 학생 모델 사이의 지식 차이를 측정하는 기존의 증류 손실 함수이다.
*   $D^T_i = L^T_{task,i}$: 교사 모델이 $i$번째 인스턴스에 대해 계산한 task loss이다. 즉, 정답과 교사 모델 예측값 사이의 거리이다.
*   $\alpha$: 하이퍼파라미터로, 가중치가 감소하는 속도를 조절한다 (실험적으로 0.1 설정).

**작동 원리**:
교사 모델의 오차 $D^T_i$가 클수록 $\exp(-\alpha D^T_i)$ 값은 작아지며, 결과적으로 학생 모델은 해당 인스턴스에 대한 교사의 지식을 덜 신뢰하게 된다. 반대로 교사 모델이 완벽하게 예측하여 $D^T_i$가 0에 가까우면 가중치는 1이 되어 교사의 지식을 최대한으로 수용한다.

### FPN 및 Self-distillation 확장
*   **Scale-wise AID**: FPN의 각 레이어마다 AID를 적용함으로써, 교사 모델이 특정 크기의 객체에 대해 더 자신감이 있는 경우에만 해당 스케일의 지식을 전달받도록 한다.
*   **Self-distillation**: 동일한 구조의 모델 두 개(Old, New)를 사용하여, 이전 모델(Old)이 잘 맞춘 데이터만 새로운 모델(New)이 학습하게 함으로써 모델 자체의 성능을 끌어올린다. 이때 손실 함수는 다음과 같다.
    $$L^{new}_i = L^{new}_{task,i} + \lambda \exp(-\alpha L^{old}_{task,i}) L^{new,old}_{distill,i}$$

## 📊 Results

### 실험 설정
*   **데이터셋**: KITTI 2D-object detection, COCO traffic 데이터셋.
*   **모델**: Two-stage (Faster R-CNN), One-stage (GFL).
*   **백본**: 교사 모델은 ResNet-101, 학생 모델은 ResNet-50을 기본으로 사용.
*   **기준선 (Baseline)**: 지식 증류를 적용하지 않은 모델(Student/Teacher-Baseline) 및 최신 어텐션 기반 KD 방법론(Zhang et al. [5]).
*   **지표**: mAP (Intersection over Union threshold = 0.5).

### 정량적 결과
KITTI 데이터셋 결과(Table I)를 보면, Zhang et al. [5]의 KD 방법만 적용했을 때보다 AID를 함께 적용했을 때 성능 향상 폭이 훨씬 컸다.
*   **Single-stage (GFL)**: Student-Baseline 대비 mAP가 최대 2.9% 상승하였다.
*   **Two-stage (Faster R-CNN)**: Student-Baseline 대비 mAP가 약 0.7% 상승하였다.
*   **Self-distillation**: 동일 백본(ResNet-101) 내에서도 AID를 통해 GFL 기준 mAP가 1.6% 상승하는 결과를 보였다.

COCO traffic 데이터셋에서도 유사하게 GFL 모델에서 teacher baseline 대비 2.5%의 mAP 향상을 달성하였다. 효율성 측면(Table III)에서는 ResNet-50 학생 모델이 ResNet-101 교사 모델 대비 파라미터 수는 평균 34.4%, FLOPs는 평균 20.4% 감소하면서도 더 높은 정확도를 보였다.

### 정성적 결과
정성적 분석(Fig. 2, 3)을 통해 AID의 효과가 입증되었다.
1.  **오답 전이 방지**: 기존 KD는 교사가 틀린 예측(예: 울타리를 자동차로 인식)을 하면 학생도 그대로 따라 하는 경향이 있었으나, AID는 교사의 오차가 큰 인스턴스의 가중치를 낮춤으로써 이러한 오류 전이를 방지하였다.
2.  **어려운 샘플 탐지**: 겹쳐 있는 객체(Overlapping objects)나 매우 작은 객체(Small-scale objects)에 대해 기존 KD보다 더 정확한 바운딩 박스를 생성하는 능력이 향상되었다.

## 🧠 Insights & Discussion

본 논문은 지식 증류 과정에서 **'무엇을(What)'** 배울 것인가만큼 **'어떤 샘플에서(Which instance)'** 배울 것인가가 중요하다는 통찰을 제시한다.

**강점**:
*   추가적인 네트워크 아키텍처 변경 없이 손실 함수에 간단한 가중치 항을 추가하는 것만으로 성능을 향상시켰다.
*   교사 모델의 신뢰도를 직접적으로 반영하는 수학적 구조($\exp(-\alpha L)$)를 통해 직관적이고 효과적인 필터링 메커니즘을 구축하였다.
*   단일/이단계 검출기 및 Self-distillation 모두에서 범용적인 성능 향상을 입증하였다.

**한계 및 논의사항**:
*   하이퍼파라미터 $\alpha$ 값에 따라 가중치 감소 폭이 결정되는데, 이에 대한 이론적 근거보다는 경험적인 튜닝($\alpha=0.1$)에 의존한 측면이 있다.
*   교사 모델의 Task Loss가 반드시 해당 인스턴스의 '지식 가치'와 일치하는지에 대한 심층적인 분석은 부족하다. 예를 들어, 교사가 틀렸지만 학생에게는 유용한 '어려운 샘플'이 존재할 수 있는데, 이를 완전히 배제하는 것이 최선인지에 대한 논의가 필요하다.

## 📌 TL;DR

본 논문은 교사 모델의 예측 오차를 기반으로 증류 가중치를 동적으로 조절하는 **Adaptive Instance Distillation (AID)** 기법을 제안한다. 교사가 잘 예측한 샘플의 지식은 많이 배우고, 확신이 없는 샘플은 학생 스스로 학습하게 함으로써, 자율 주행 객체 탐지 모델의 정확도와 효율성을 동시에 높였다. 이 방법은 특히 작은 객체나 겹친 객체 탐지 성능을 개선하며, 다양한 검출기 구조와 Self-distillation에도 적용 가능하다는 점에서 실용적 가치가 높다.