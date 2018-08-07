
def is_subset(which, of_what):
	return 0 < len(which & of_what) <= len(of_what)

def is_identical(set1, set2):
	return len(set1 & set2) == len(set1) == len(set2)

def has_no_overlap(set1, set2):
	return len(set1 & set2) == 0