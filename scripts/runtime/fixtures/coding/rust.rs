pub fn bounded_append<T>(items: &mut Vec<T>, value: T, capacity: usize) -> Result<(), &'static str> {
    if items.len() >= capacity {
        return Err("capacity exceeded");
    }
    items.push(value);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_overflow_without_mutation() {
        let mut items = vec![1];
        assert_eq!(bounded_append(&mut items, 2, 1), Err("capacity exceeded"));
        assert_eq!(items, vec![1]);
    }
}
