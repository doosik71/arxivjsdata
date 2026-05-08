# Contour Loss for Instance Segmentation via k-step Distance Transformation Image

Xiaolong Guo, Xiaosong Lan, Kunfeng Wang, Shuxiao Li

## 🧩 Problem to Solve

기존 인스턴스 분할(Instance Segmentation) 방법, 특히 Mask R-CNN과 같은 대표적인 모델은 픽셀 단위의 마스크를 생성하지만, 예측된 마스크의 윤곽선(contour) 부분이 불분명하고 정확하지 않다는 문제가 있습니다. 이는 로봇 그랩(robot grabbing)과 같은 정밀한 윤곽선 정보가 필수적인 응용 분야에서 치명적인 결함이 될 수 있습니다. 이 논문은 예측된 마스크와 실제(ground-truth) 마스크가 전체적으로 일치할 뿐만 아니라, 특히 윤곽선 근처에서 최대한 일치하도록 만드는 것을 목표로 합니다.

## ✨ Key Contributions

* **새로운 손실 함수 'Contour Loss' 제안:** 거리 변환 이미지(Distance Transformation Image, DTI)의 개념을 인스턴스 분할에 도입하여 예측 마스크의 윤곽선 부분을 특별히 최적화하는 새로운 손실 함수를 제안했습니다.
* **미분 가능한 $k$-step DTI 계산 모듈 설계:** 현대 신경망 프레임워크에서 윤곽선 손실을 공동 학습할 수 있도록 예측 마스크와 실제 마스크의 잘린 DTI(truncated DTI)를 온라인으로 근사 계산하는 미분 가능한 $k$-step DTI 모듈(`kSDT`)을 설계했습니다. 이는 현재 신경망 프레임워크에 적응 가능한 최초의 분석적 DTI 모듈입니다.
* **높은 범용성 및 효과 검증:** 제안된 Contour Loss는 Mask R-CNN과 같은 기존 인스턴스 분할 방법의 추론 네트워크 구조를 수정하거나 추가 학습 파라미터를 늘리지 않고도 기존 손실 함수와 쉽게 통합될 수 있어 높은 범용성을 가집니다. COCO 데이터셋 실험을 통해 더 정확하고 선명한 마스크를 생성하며, 인스턴스 분할 성능을 향상시키는 데 효과적임을 입증했습니다.

## 📎 Related Works

이 논문은 인스턴스 분할 분야의 주요 연구들과 윤곽선(edge) 또는 경계(boundary) 정보를 활용하는 방법들을 참조하고 차별점을 설명합니다.

* **주류 인스턴스 분할 방법:**
  * **분할 기반(Segmentation-based) 방법:** FCN [8]을 시작으로 Instance-FCN [9], FCIS [10], Biseg [11], SOLO [12] 등이 있습니다.
  * **탐지 기반(Detection-based) 방법:** Mask R-CNN [2]이 대표적이며, RetinaMask [15], PA-Net [16], Mask Scoring R-CNN [17], HTC [18] 등이 있습니다. 본 논문은 탐지 기반 방법인 Mask R-CNN을 기본 프레임워크로 채택했습니다.
* **경계 정보 결합 방법:**
  * Kang et al. [19]: 실제 마스크의 윤곽선을 $k$ 픽셀만큼 확장하여 풍부한 엣지 정보를 학습하려 했으나, 하이퍼파라미터 $k$에 민감하며 이론적 기반이 부족합니다. 본 논문의 방법은 고전적인 DTI 기술에 기반하며 $k$에 덜 민감합니다.
  * Roland et al. [20]: Sobel [21] 연산자를 사용하여 엣지 이미지를 추출하고 MSE(Mean Square Error) 손실로 오류를 측정했습니다. 본 논문의 방법은 단순 위치 정보 외에 거리 정보를 인코딩하는 $k$-step DTI를 사용하여 객체 윤곽을 더 잘 학습합니다.
  * Hayder et al. [22]: DTI를 마스크 표현으로 사용하고 복잡한 신경망 분기를 통해 예측했습니다. 이 방법은 윤곽선에서 멀리 떨어진 영역의 DTI 값이 불안정할 수 있으며, 복잡한 파이프라인과 후처리 단계가 필요합니다. 본 논문의 방법은 잘린(truncated) DTI 모듈을 사용하여 윤곽점 최적화에 집중하고, 기존 추론 네트워크 구조를 유지하며 추가 후처리가 필요 없습니다.
  * Cheng et al. [23]: 마스크 엣지 예측을 위한 새로운 분기(branch)를 학습시켜 훈련 파라미터를 직접 증가시켰습니다. 본 논문의 방법은 기본 알고리즘의 네트워크 구조를 수정하거나 훈련 파라미터를 증가시키지 않고 기존 파라미터만 최적화합니다.

## 🛠️ Methodology

제안하는 Contour Loss는 Mask R-CNN과 같은 기존 인스턴스 분할 프레임워크에 통합되어 공동 학습됩니다. 그 계산 과정은 다음과 같습니다:

1. **예측 마스크 반응 이진화($B(\cdot)$):**
    * 마스크 분기(mask branch)의 출력에서 선택된 예측 마스크 반응 $M_{R}$을 미분 가능한 함수를 사용하여 근사적으로 이진화하여 예측 마스크 $M_{P}$를 얻습니다.
    * $B(x) = \frac{1}{1 + e^{-\gamma(x-T)}}$
    * $M_{P} = B(M_{R})$
    * 여기서 $\gamma$는 기울기, $T$는 임계값이며, 기본값은 $\gamma=20$, $T=0.5$입니다.

2. **윤곽선 반응 계산(`ConvSobel(·)`):**
    * 고정된 Sobel 연산자(Sobel$_{x}$, Sobel$_{y}$)를 합성곱 커널로 사용하는 합성곱 계층을 구성하여 예측 마스크 $M_{P}$와 실제 마스크 $M_{GT}$에 적용하여 각각 예측 윤곽선 반응 $\Omega_{PCR}$과 실제 윤곽선 반응 $\Omega_{GCR}$을 얻습니다.
    * $\Omega_{PCR} = \frac{1}{2} [|M_{P} * \text{Sobel}_{x}| + |M_{P} * \text{Sobel}_{y}|]$
    * $\Omega_{GCR} = \frac{1}{2} [|M_{GT} * \text{Sobel}_{x}| + |M_{GT} * \text{Sobel}_{y}|]$

3. **$k$-step DTI 계산(`kSDT(·)`):**
    * 일반적인 DTI는 윤곽선에서 멀리 떨어진 픽셀 값이 불안정할 수 있어 최적화를 방해할 수 있습니다. 윤곽선 최적화에 집중하기 위해 임계값 $k$를 사용하여 DTI를 잘라냅니다($k$-step DTI). $k$를 초과하는 픽셀 값은 $k$로 설정됩니다.
    * 이를 위해 미분 가능한 근사 $k$-step DTI 모듈(`kSDT`)을 설계했습니다.
    * $kSDT$는 윤곽선 반응 이미지를 초기 입력으로 받아 $k$단계의 반복 연산을 수행합니다:
        * `one-step dilation` 연산 $D(\cdot)$ (수식 (9))
        * `element-wise addition` 연산 $\oplus$ (수식 (10))
    * 미분 가능한 `one-step dilation` 연산은 `Smooth` 연산자(1/9 커널)와의 합성곱 후 이진화 함수 $B(\cdot)$ (여기서 $\gamma=20$, $T=0.1$)를 사용하여 구현됩니다.
    * $\Gamma_{k,P} = kSDT(\Omega_{PCR})$
    * $\Gamma_{k,GT} = kSDT(\Omega_{GCR})$

4. **윤곽선 손실 함수 계산($L_{Contour}$):**
    * 예측 윤곽선과 실제 윤곽선 간의 거리를 측정합니다. 하나의 윤곽선 이미지의 $k$-step DTI에 대한 다른 윤곽선 이미지의 커버리지 값을 누적합니다.
    * 이를 미분 가능하게 하기 위해 연속 버전의 수식을 사용합니다.
    * $d(\Omega_{PCR}, \Omega_{GCR}) = \frac{1}{2} \left[ \frac{\text{GAP}(\Omega_{PCR} \otimes \Gamma_{k,GT})}{\text{GAP}(\Omega_{PCR}) + \epsilon} + \frac{\text{GAP}(\Omega_{GCR} \otimes \Gamma_{k,P})}{\text{GAP}(\Omega_{GCR}) + \epsilon} \right]$
        * $\otimes$는 Hadamard 곱(element-wise product), $\text{GAP}(\cdot)$는 전역 평균 풀링(Global Average Pooling), $\epsilon$은 0 나눔을 방지하는 스무딩 항입니다.
    * 최종 윤곽선 손실은 학습 배치 내의 모든 양성 샘플에 대한 이 값의 평균입니다:
    * $L_{Contour} = \frac{1}{N} \sum_{i=1}^{N} d(\Omega_{i,PCR}, \Omega_{i,GCR})$

5. **공동 학습:**
    * 전체 손실은 기존 Mask R-CNN의 분류 손실($L_{cls}$), 바운딩 박스 회귀 손실($L_{box}$), 마스크 손실($L_{mask}$)과 제안된 윤곽선 손실($L_{Contour}$)을 결합한 다중 작업 손실로 정의됩니다.
    * $L = L_{cls} + L_{box} + L_{mask} + L_{Contour}$
    * Contour Loss는 기본 Mask R-CNN이 어느 정도 학습된 후(예: 120K 반복 후) 활성화되어 추가 학습됩니다.

## 📊 Results

COCO 2014 데이터셋(82783 훈련 이미지, 40504 검증 이미지 중 5000개 서브셋 사용)에서 Mask R-CNN을 기반으로 광범위한 실험을 수행했습니다. 평가지표는 COCO AP(mAP, AP50, AP75, APs, APm, APl)를 사용했습니다.

* **하이퍼파라미터 $k$ 평가:**
  * $k$ 값(1, 2, 3, 4, 5, 6)을 실험한 결과, $k=2$일 때 가장 높은 성능 향상을 보였습니다.
  * $k=2$에서 기준선 대비 mAP 0.26%, AP50 0.22%, AP75 0.34%, APs 0.44%, APm 0.2%, APl 0.58% 향상되었습니다. $k$는 1-5 범위에서 성능에 크게 민감하지 않았습니다.
* **Ablation Study (다른 손실 함수와의 비교):**
  * `MSE Edge Loss` (예측/실제 윤곽선 반응 간 MSE) 및 `MSE Contour Loss` (예측/실제 $k$-step DTI 간 MSE)와 비교했습니다.
  * 모든 제안된 손실 함수가 기준선 대비 마스크 정확도를 향상시켰으며, `Contour Loss`가 가장 우수한 마스크 정확도를 달성했습니다.
* **Comparative Study (다른 모델 및 백본에서의 일반화 능력):**
  * Mask R-CNN에 Res-50+FPN, Res-101+FPN, Res-X-101 백본을 사용했을 때, `Contour Loss`는 각각 0.13%~0.26% mAP의 성능 향상을 가져왔습니다.
  * HTC (Res-50+FPN 백본)에서도 mAP 0.2% 향상을 보여, 다양한 인스턴스 분할 방법 및 백본에 대해 효과적인 일반화 능력을 입증했습니다.
* **정성적 분석:**
  * Mask R-CNN 단독 결과와 `Mask R-CNN + Contour Loss` 결과를 시각적으로 비교했을 때, 제안된 방법이 훨씬 더 정확하고 선명한 객체 윤곽선을 생성함을 확인했습니다.

## 🧠 Insights & Discussion

* 이 논문은 인스턴스 분할에서 윤곽선 정확도 문제를 해결하기 위해 DTI라는 고전적인 이미지 처리 기법을 딥러닝 프레임워크에 성공적으로 통합했습니다.
* 핵심 아이디어인 미분 가능한 $k$-step DTI 모듈(`kSDT`)은 DTI의 비미분성 문제를 극복하고, 윤곽선 부분에 대한 명시적인 최적화를 가능하게 함으로써 마스크의 품질을 크게 향상시켰습니다.
* 제안된 Contour Loss는 기존 네트워크 구조 변경이나 추가 파라미터 없이 인스턴스 분할 모델의 성능을 향상시킬 수 있어 매우 실용적입니다. 이는 로봇 비전과 같이 정확한 윤곽선이 중요한 실제 응용 분야에 큰 이점을 제공합니다.
* 연구의 한계점으로는 미확인 객체(unseen objects)의 인스턴스 분할에 Contour Loss를 적용하는 가능성에 대한 탐색이 향후 과제로 언급되었습니다.

## 📌 TL;DR

인스턴스 분할 모델, 특히 Mask R-CNN이 예측하는 마스크의 윤곽선이 불분명하고 부정확한 문제를 해결하기 위해, 이 논문은 거리 변환 이미지(DTI)를 기반으로 하는 새로운 손실 함수인 **Contour Loss**를 제안합니다. 이 손실 함수는 예측 마스크의 윤곽선 부분을 특별히 최적화하며, 이를 위해 미분 가능한 $k$-step DTI 계산 모듈(`kSDT`)을 개발하여 기존 신경망 프레임워크에서 온라인으로 공동 학습이 가능하도록 했습니다. COCO 데이터셋 실험을 통해 Contour Loss가 Mask R-CNN과 HTC 등 다양한 모델의 윤곽선 정확도를 포함한 전반적인 인스턴스 분할 성능을 효과적으로 향상시키며, 네트워크 구조 변경이나 추가 파라미터 없이 쉽게 통합될 수 있음을 입증했습니다.
