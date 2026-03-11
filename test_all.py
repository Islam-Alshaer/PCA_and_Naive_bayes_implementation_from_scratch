#!/usr/bin/env python3
"""Test all experiments"""
import sys
import traceback

def test_experiments():
    """Test all experiments"""
    import json
    import numpy as np
    import pandas as pd
    from sklearn.feature_selection import SelectKBest, chi2
    from naive_bayes_gaussian import naive_bayes_gaussian
    from naive_bayes_categorial import NaiveBayesCategorial
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
    from my_PCA import my_PCA

    output = []
    json_results = []
    tests_passed = 0
    tests_failed = 0

    # Test 1-0
    try:
        msg = "[TEST 1-0] Categorical NB on car_evaluation... "
        car_evaluation = pd.read_csv('car_evaluation.csv')
        X = car_evaluation.drop('class', axis=1)
        y = car_evaluation['class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        categorialNB = NaiveBayesCategorial()
        categorialNB.fit(X_train, y_train)
        y_pred = categorialNB.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        msg += f"✓ PASSED (accuracy: {acc:.3f})"
        output.append(msg)
        json_results.append({
            'test': '1-0',
            'model': 'NaiveBayesCategorial',
            'dataset': 'car_evaluation',
            'feature_selection': 'none',
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1)
        })
        tests_passed += 1
    except Exception as e:
        msg += f"✗ FAILED: {str(e)[:100]}"
        output.append(msg)
        output.append(traceback.format_exc())
        tests_failed += 1

    # Test 1-A
    try:
        msg = "[TEST 1-A] Categorical NB with feature selection... "
        ordinal_enc = OrdinalEncoder()
        X_train_ordinal = ordinal_enc.fit_transform(X_train)
        y_train_labeled = LabelEncoder().fit_transform(y_train)
        selector = SelectKBest(chi2, k=4)
        X_new = selector.fit_transform(X_train_ordinal, y_train_labeled)
        mask = selector.get_support()
        selected_features = X_train.columns[mask]
        X_train_filtered = X_train[selected_features]
        X_test_filtered = X_test[selected_features]
        categorialNB.fit(X_train_filtered, y_train)
        y_pred_new = categorialNB.predict(X_test_filtered)
        acc = accuracy_score(y_test, y_pred_new)
        prec = precision_score(y_test, y_pred_new, average='weighted')
        rec = recall_score(y_test, y_pred_new, average='weighted')
        f1 = f1_score(y_test, y_pred_new, average='weighted')
        msg += f"✓ PASSED (accuracy: {acc:.3f})"
        output.append(msg)
        json_results.append({
            'test': '1-A',
            'model': 'NaiveBayesCategorial',
            'dataset': 'car_evaluation',
            'feature_selection': 'SelectKBest(chi2, k=4)',
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1)
        })
        tests_passed += 1
    except Exception as e:
        msg += f"✗ FAILED: {str(e)[:100]}"
        output.append(msg)
        output.append(traceback.format_exc())
        tests_failed += 1

    # Test 1-B
    try:
        msg = "[TEST 1-B] Gaussian NB with PCA... "
        pca = my_PCA(n_components=4)
        ordinal_enc = OrdinalEncoder()
        X_train_ordinal = ordinal_enc.fit_transform(X_train)
        X_test_ordinal = ordinal_enc.transform(X_test)
        pca.fit(X_train_ordinal)
        X_train_PCA = np.asarray(pca.transform(X_train_ordinal))
        X_test_PCA = np.asarray(pca.transform(X_test_ordinal))
        gaussianNB = naive_bayes_gaussian()
        gaussianNB.fit(X_train_PCA, y_train)
        y_pred_new = gaussianNB.predict(X_test_PCA)
        acc = accuracy_score(y_test, y_pred_new)
        prec = precision_score(y_test, y_pred_new, average='weighted')
        rec = recall_score(y_test, y_pred_new, average='weighted')
        f1 = f1_score(y_test, y_pred_new, average='weighted')
        msg += f"✓ PASSED (accuracy: {acc:.3f})"
        output.append(msg)
        json_results.append({
            'test': '1-B',
            'model': 'naive_bayes_gaussian',
            'dataset': 'car_evaluation',
            'feature_selection': 'PCA(n_components=4)',
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1)
        })
        tests_passed += 1
    except Exception as e:
        msg += f"✗ FAILED: {str(e)[:100]}"
        output.append(msg)
        output.append(traceback.format_exc())
        tests_failed += 1

    # Test 2-0
    try:
        msg = "[TEST 2-0] Gaussian NB on student_placement... "
        numericalNB = naive_bayes_gaussian()
        student_placement = pd.read_csv('student_placement_synthetic.csv')
        # Encode categorical columns
        le_dict = {}
        for col in student_placement.columns:
            le = LabelEncoder()
            student_placement[col] = le.fit_transform(student_placement[col].astype(str))
            le_dict[col] = le
        X = student_placement.drop('placement_status', axis=1).astype(float)
        y = student_placement['placement_status'].astype(float)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        numericalNB.fit(X_train, y_train)
        y_pred = numericalNB.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        msg += f"✓ PASSED (accuracy: {acc:.3f})"
        output.append(msg)
        json_results.append({
            'test': '2-0',
            'model': 'naive_bayes_gaussian',
            'dataset': 'student_placement',
            'feature_selection': 'none',
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1)
        })
        tests_passed += 1
    except Exception as e:
        msg += f"✗ FAILED: {str(e)[:100]}"
        output.append(msg)
        output.append(traceback.format_exc())
        tests_failed += 1

    # Test 2-A
    try:
        msg = "[TEST 2-A] Gaussian NB with feature selection... "
        selector = SelectKBest(chi2, k=9)
        X_new = selector.fit_transform(X_train, y_train)
        X_test_new = selector.transform(X_test)
        numericalNB.fit(X_new, y_train)
        y_pred_new = numericalNB.predict(X_test_new)
        acc = accuracy_score(y_test, y_pred_new)
        prec = precision_score(y_test, y_pred_new, average='weighted')
        rec = recall_score(y_test, y_pred_new, average='weighted')
        f1 = f1_score(y_test, y_pred_new, average='weighted')
        msg += f"✓ PASSED (accuracy: {acc:.3f})"
        output.append(msg)
        json_results.append({
            'test': '2-A',
            'model': 'naive_bayes_gaussian',
            'dataset': 'student_placement',
            'feature_selection': 'SelectKBest(chi2, k=9)',
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1)
        })
        tests_passed += 1
    except Exception as e:
        msg += f"✗ FAILED: {str(e)[:100]}"
        output.append(msg)
        output.append(traceback.format_exc())
        tests_failed += 1

    # Test 2-B
    try:
        msg = "[TEST 2-B] Gaussian NB with PCA (9 components)... "
        pca = my_PCA(n_components=9)
        pca.fit(X_train)
        X_new = np.asarray(pca.transform(X_train))
        X_test_new = np.asarray(pca.transform(X_test))
        numericalNB.fit(X_new, y_train)
        y_pred_new = numericalNB.predict(X_test_new)
        acc = accuracy_score(y_test, y_pred_new)
        prec = precision_score(y_test, y_pred_new, average='weighted')
        rec = recall_score(y_test, y_pred_new, average='weighted')
        f1 = f1_score(y_test, y_pred_new, average='weighted')
        msg += f"✓ PASSED (accuracy: {acc:.3f})"
        output.append(msg)
        json_results.append({
            'test': '2-B',
            'model': 'naive_bayes_gaussian',
            'dataset': 'student_placement',
            'feature_selection': 'PCA(n_components=9)',
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1)
        })
        tests_passed += 1
    except Exception as e:
        msg += f"✗ FAILED: {str(e)[:100]}"
        output.append(msg)
        output.append(traceback.format_exc())
        tests_failed += 1

    output.append(f"\n{'='*60}")
    output.append(f"RESULTS: {tests_passed} passed, {tests_failed} failed")
    output.append(f"{'='*60}")

    # Write to both stdout and file
    output_str = '\n'.join(output)
    print(output_str)
    with open('test_results.txt', 'w') as f:
        f.write(output_str)

    # Write JSON results to file
    with open('test_results.json', 'w') as f:
        json.dump(json_results, f, indent=4)

    output.append(f"\n✓ JSON results saved to 'test_results.json'")

    return tests_failed == 0

if __name__ == '__main__':
    success = test_experiments()
    sys.exit(0 if success else 1)

