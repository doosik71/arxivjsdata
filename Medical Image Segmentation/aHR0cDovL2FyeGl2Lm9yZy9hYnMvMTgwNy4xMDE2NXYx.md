# UNet++: A Nested U-Net Architecture for Medical Image Segmentation

Zongwei Zhou, Md Mahfuzur Rahman Siddiquee, Nima Tajbakhsh, and Jianming Liang (2018)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 의료 영상 분할(Medical Image Segmentation)에서 인코더(Encoder)와 디코더(Decoder) 사이의 **semantic gap(의미론적 차이)**으로 인해 발생하는 정밀도 저하이다.

일반적인 U-Net과 같은 Encoder-Decoder 구조는 Skip connection을 통해 인코더의 저수준(low-level) 특징 맵을 디코더의 고수준(high-level) 특징 맵과 직접 결합한다. 그러나 의료 영상에서는 작은 결절의 침상 패턴(spiculation patterns)과 같은 매우 세밀한 디테일이 진단에 결정적인 영향을 미치므로, 단순히 서로 다른 수준의 특징 맵을 결합하는 방식으로는 충분한 정확도를 얻기 어렵다. 특히, 세그멘테이션 마스크의 미세한 오류가 임상 환경에서는 잘못된 진단이나 치료 계획으로 이어질 수 있다는 점에서 문제의 중요성이 매우 높다. 따라서 본 논문의 목표는 인코더의 고해상도 특징 맵을 디코더에 융합하기 전 점진적으로 풍부하게 만들어, 두 네트워크 간의 semantic gap을 줄임으로써 더 정밀한 분할 성능을 달성하는 것이다.

## ✨ Key Contributions

본 논문의 중심적인 설계 아이디어는 **Nested and Dense Skip Pathways**와 **Deep Supervision**의 도입이다.

단순히 특징 맵을 전달하는 기존의 Skip connection 대신, 인코더와 디코더 사이에 중첩된(nested) 밀집 컨볼루션 블록(dense convolution blocks)을 배치하여 특징 맵을 단계적으로 처리한다. 이를 통해 인코더의 특징 맵이 디코더의 특징 맵과 유사한 의미론적 수준을 갖게 하여 최적화 과정에서의 학습 난이도를 낮춘다. 또한, 여러 단계의 출력단에서 손실 함수를 계산하는 Deep supervision을 적용하여 모델의 학습을 돕고, 추론 시에는 필요에 따라 네트워크의 일부를 제거하는 Pruning(가지치기)을 통해 속도와 정확도 사이의 트레이드오프를 조절할 수 있게 설계하였다.

## 📎 Related Works

기존의 이미지 분할 모델인 FCN과 U-Net은 모두 Skip connection을 사용하여 해상도를 복구하고 세밀한 디테일을 살리는 방식을 취하고 있다. FCN은 업샘플링된 특징 맵을 인코더의 특징 맵과 합산(sum)하고, U-Net은 이를 연결(concatenate)한 후 컨볼루션을 적용한다. 또한, DenseNet의 구조를 차용한 H-denseunet이나 특징 맵을 그리드 형태로 연결한 GridNet 등이 제안된 바 있다.

하지만 본 논문은 이러한 기존 접근 방식들이 인코더와 디코더의 특징 맵이 서로 의미론적으로 매우 다름에도 불구하고 이를 직접 융합한다는 점에 주목한다. 이러한 성질이 결과적으로 분할 성능을 저하시킬 수 있다고 주장하며, 이를 해결하기 위해 특징 맵의 의미 수준을 점진적으로 맞추는 중첩 구조를 제안함으로써 차별성을 갖는다.

## 🛠️ Methodology

### 전체 시스템 구조
UNet++는 기본적으로 Encoder-Decoder 구조를 유지하지만, 그 사이를 연결하는 Skip pathway가 재설계되었다. 인코더에서 추출된 특징 맵은 디코더로 바로 전달되지 않고, 피라미드 레벨에 따라 결정되는 개수의 컨볼루션 레이어로 구성된 Dense convolution block을 통과한다.

### 재설계된 Skip Pathways
Skip pathway의 핵심은 인코더의 특징 맵을 점진적으로 정제하는 것이다. 예를 들어, 최상단 경로에서는 여러 개의 컨볼루션 레이어가 중첩되어 있으며, 각 레이어는 이전 레이어의 출력과 하단 경로에서 업샘플링되어 올라온 특징 맵을 모두 입력으로 받아 결합한다.

이를 수식으로 표현하면 다음과 같다. $x_{i,j}$를 노드 $X_{i,j}$의 출력이라고 할 때, $i$는 인코더의 다운샘플링 레벨을, $j$는 Skip pathway 내의 컨볼루션 레이어 인덱스를 나타낸다.

$$
x_{i,j} = 
\begin{cases} 
H(x_{i-1,j}), & j= 0 \\
H([ [x_{i,k}]_{k=0}^{j-1}, U(x_{i+1,j-1}) ]), & j > 0 
\end{cases}
$$

여기서 $H(\cdot)$는 컨볼루션 연산과 활성화 함수를 의미하며, $U(\cdot)$는 업샘플링 레이어, $[ \ ]$는 연결(concatenation) 연산을 의미한다. 즉, $j=0$인 노드는 인코더에서 직접 입력을 받고, $j>0$인 노드는 동일 경로의 이전 모든 노드들의 출력과 하단 레벨의 업샘플링된 출력을 모두 합쳐서 처리함으로써 특징 맵을 점진적으로 풍부하게 만든다.

### Deep Supervision 및 손실 함수
UNet++는 중첩 구조 덕분에 여러 개의 해상도 특징 맵 $\{x_{0,j} \mid j \in \{1, 2, 3, 4\}\}$을 생성할 수 있다. 각 출력단에 $1 \times 1$ 컨볼루션과 시그모이드 함수를 적용하여 4개의 세그멘테이션 맵을 만들고, 이를 위해 다음과 같은 손실 함수를 사용한다.

$$
L(Y, \hat{Y}) = -\frac{1}{N} \sum_{b=1}^{N} \left( \frac{1}{2} \cdot Y_b \cdot \log \hat{Y}_b + \frac{2 \cdot Y_b \cdot \hat{Y}_b}{Y_b + \hat{Y}_b} \right)
$$

이 식은 이진 교차 엔트로피(Binary Cross-Entropy)와 다이스 계수(Dice Coefficient)를 결합한 형태로, $Y_b$는 정답(ground truth), $\hat{Y}_b$는 예측 확률을 의미한다.

### 추론 모드 및 Pruning
Deep supervision을 통해 학습된 모델은 두 가지 모드로 작동할 수 있다.
1. **Accurate mode**: 모든 세그멘테이션 브랜치의 출력을 평균 내어 최종 결과를 생성한다.
2. **Fast mode**: 특정 브랜치의 출력 하나만을 선택하여 결과로 사용한다. 이를 통해 네트워크의 일부를 제거(pruning)함으로써 추론 속도를 비약적으로 높일 수 있다.

## 📊 Results

### 실험 설정
- **데이터셋**: 세포 핵(cell nuclei), 결장 폴립(colon polyp), 간(liver), 폐 결절(lung nodule) 등 4가지 의료 영상 데이터셋을 사용하였다.
- **비교 대상**: 기본 U-Net 및 파라미터 수를 UNet++와 유사하게 맞춘 Wide U-Net을 기준선으로 설정하였다. 이는 성능 향상이 단순히 파라미터 증가에 의한 것이 아님을 증명하기 위함이다.
- **평가 지표**: Dice coefficient와 Intersection over Union (IoU)를 측정하였다.

### 정량적 결과
실험 결과, Deep supervision을 적용한 UNet++가 U-Net 대비 평균 3.9포인트, Wide U-Net 대비 평균 3.4포인트의 IoU 이득을 얻었다. 특히 Wide U-Net가 U-Net보다 전반적으로 높은 성능을 보였음에도 불구하고, UNet++는 이를 상회하는 성능을 기록하였다. Deep supervision의 효과는 특히 간(liver)과 폐 결절(lung nodule) 데이터셋에서 뚜렷하게 나타났는데, 이는 해당 객체들이 영상 내에서 다양한 크기로 나타나므로 다중 스케일 접근 방식이 필수적이기 때문이다.

### 모델 Pruning 결과
추론 속도 분석 결과, $\text{UNet}++ L_3$ 수준으로 프루닝을 진행했을 때 IoU는 단 0.6포인트만 하락하면서도 추론 시간은 평균 32.2% 단축되는 효율성을 보였다.

## 🧠 Insights & Discussion

본 논문은 인코더와 디코더 사이의 semantic gap을 줄이는 것이 의료 영상 분할의 정확도를 높이는 핵심임을 입증하였다. 특히 단순한 네트워크 깊이 증가나 파라미터 확장이 아닌, 특징 맵의 연결 구조를 개선함으로써 최적화 문제를 더 쉽게 만들었다는 점이 강점이다.

또한, Deep supervision을 통해 정확도 향상뿐만 아니라 추론 시의 유연성(Pruning을 통한 속도 조절)을 확보한 점은 실제 의료 현장에서의 하드웨어 제약 사항을 고려했을 때 매우 실용적인 접근이다. 다만, 모든 데이터셋에서 Deep supervision의 효과가 동일하게 나타나지 않았다는 점(예: 세포 핵 데이터셋)은 객체의 특성이나 데이터셋의 복잡도에 따라 최적의 구조가 다를 수 있음을 시사한다.

## 📌 TL;DR

본 논문은 U-Net의 Skip connection을 중첩된 밀집 구조(Nested Dense Skip Pathways)로 재설계하여 인코더와 디코더 간의 의미론적 차이를 줄인 **UNet++**를 제안한다. Deep supervision을 통해 학습 효율과 다중 스케일 대응 능력을 높였으며, 추론 시에는 모델 Pruning을 통해 속도를 최적화할 수 있다. 실험 결과, 기존 U-Net 및 Wide U-Net 대비 유의미한 IoU 향상을 달성하여 정밀한 의료 영상 분할에 매우 효과적임을 입증하였다.