
// boolean type which supports any value

use std::fmt;
use std::ops;

use bytemuck::{Pod, Zeroable};
use num;

#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub struct Bool(pub u8);

// SAFETY: Bool layout is identical to u8 layout, which is Pod.
// Any bit pattern is valid, unlike Rust's native `bool` type.
unsafe impl Zeroable for Bool {}
unsafe impl Pod for Bool {}

impl Bool {
    pub fn canonicalize(self) -> bool { self.0 != 0 }
}

impl PartialEq for Bool {
    fn eq(&self, other: &Self) -> bool { self.canonicalize() == other.canonicalize() }
}

impl Eq for Bool { }

impl PartialOrd for Bool {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.canonicalize().partial_cmp(&other.canonicalize())
    }
}

impl fmt::Display for Bool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.canonicalize().fmt(f)
    }
}

impl From<u8> for Bool {
    fn from(value: u8) -> Self { Bool(value) }
}

impl From<bool> for Bool {
    fn from(value: bool) -> Self { Bool(value as u8) }
}

impl From<Bool> for bool {
    fn from(value: Bool) -> Self { value.canonicalize() }
}

impl ops::BitAnd<Bool> for Bool {
    type Output = Self;
    fn bitand(self, rhs: Bool) -> Self::Output { Self(self.0 & rhs.0) }
}

impl ops::BitOr<Bool> for Bool {
    type Output = Self;
    fn bitor(self, rhs: Bool) -> Self::Output { Self(self.0 | rhs.0) }
}

impl ops::BitXor<Bool> for Bool {
    type Output = Self;
    fn bitxor(self, rhs: Bool) -> Self::Output { Self(self.0 ^ rhs.0) }
}

impl ops::Not for Bool {
    type Output = Self;
    fn not(self) -> Self::Output { Self(!self.0) }
}

impl ops::Add<Bool> for Bool {
    type Output = Self;
    fn add(self, rhs: Bool) -> Self::Output { self | rhs }
}

impl ops::Sub<Bool> for Bool {
    type Output = Self;
    fn sub(self, rhs: Bool) -> Self::Output { self & !rhs }
}

impl ops::Mul<Bool> for Bool {
    type Output = Self;
    fn mul(self, rhs: Bool) -> Self::Output { self & rhs }
}

impl ops::Div<Bool> for Bool {
    type Output = Self;
    fn div(self, rhs: Bool) -> Self::Output { Self(self.0 / rhs.0) }
}

impl ops::Rem<Bool> for Bool {
    type Output = Self;
    fn rem(self, rhs: Bool) -> Self::Output { Self(self.0 % rhs.0) }
}

impl ops::BitAndAssign<Bool> for Bool {
    fn bitand_assign(&mut self, rhs: Bool) { self.0 &= rhs.0; }
}

impl ops::BitOrAssign<Bool> for Bool {
    fn bitor_assign(&mut self, rhs: Bool) { self.0 |= rhs.0; }
}

impl ops::BitXorAssign<Bool> for Bool {
    fn bitxor_assign(&mut self, rhs: Bool) { self.0 ^= rhs.0; }
}

impl ops::AddAssign<Bool> for Bool {
    fn add_assign(&mut self, rhs: Bool) { *self |= rhs; }
}

impl ops::SubAssign<Bool> for Bool {
    fn sub_assign(&mut self, rhs: Bool) { *self &= !rhs; }
}

impl ops::MulAssign<Bool> for Bool {
    fn mul_assign(&mut self, rhs: Bool) { *self &= rhs; }
}

impl num::Zero for Bool {
    fn zero() -> Self { Bool(0u8) }

    fn is_zero(&self) -> bool { !self.canonicalize() }
}

impl num::One for Bool {
    fn one() -> Self { Bool(1u8) }

    fn is_one(&self) -> bool { self.canonicalize() }
}

impl num::Num for Bool {
    type FromStrRadixErr = <u8 as num::Num>::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        u8::from_str_radix(str, radix).map(Bool::from)
    }
}

impl num::Unsigned for Bool { }