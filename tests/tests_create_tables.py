import re

def test_truncate():
	p = re.compile(ur'[^_][^_]+(?=.csv)')
	test_str = u"20160812_experiment_run_object.csv"
	out = re.search(p, test_str).group()
	assert out == 'object'