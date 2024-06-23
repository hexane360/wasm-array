#![feature(iter_intersperse)]

use proc_macro;

use syn::parse::{Parse, ParseStream, Result};
use syn::{parse_macro_input, parse_quote, Error, Expr, ExprClosure, Ident, Pat, PatIdent, Type, TypeTuple};
use syn::{Item, ItemImpl, ImplItem, ImplItemFn, Generics, GenericParam, Block, FnArg};
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::token::Comma;
use quote::{quote, quote_spanned};

struct TypeDispatches {
    dtype: Option<Ident>,
    inner: Punctuated<TypeDispatch, Comma>,
}

impl Parse for TypeDispatches {
    fn parse(input: ParseStream) -> Result<Self> {
        let dtype = Ident::parse(input).ok();
        Ok(Self { dtype, inner: Punctuated::parse_terminated(input)? })
    }
}

struct TypeDispatch {
    types: TypeTuple,
    _comma1: Comma,
    closure: ExprClosure,
    idents: Vec<PatIdent>,
}

impl Parse for TypeDispatch {
    fn parse(input: ParseStream) -> Result<Self> {
        let types = input.parse()?;
        let _comma1 = input.parse()?;
        let closure: ExprClosure = input.parse()?;

        /*if closure.inputs.len() == 0 {
            return Err(Error::new(closure.inputs.span(), "expected one or more bindings"));
        }*/

        let mut idents = Vec::with_capacity(closure.inputs.len());

        for pat in closure.inputs.iter() {
            if let Pat::Ident(ident) = pat {
                idents.push(ident.to_owned());
            } else {
                return Err(Error::new(pat.span(), "expected pattern ident binding"));
            }
        }

        Ok(Self {
            types, _comma1, closure, idents,
        })
    }
}

#[proc_macro]
pub fn type_dispatch(tokens: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(tokens as TypeDispatches);

    let first_ident: Option<Ident> = input.inner.first().and_then(|i| i.idents.first()).map(|i| i.ident.to_owned());

    let mut arms: Vec<_> = input.inner.into_iter().flat_map(|input| {
        input.types.elems.into_iter().map(move |ty: Type| {
            let let_stmts: Vec<_> = input.idents.iter().map(|binding| match (binding.by_ref, binding.mutability, &binding.ident) {
                (None, _, ident) => quote! { let #ident = #ident.downcast::<#ty>().unwrap(); },
                (Some(_), None, ident) => quote! { let #ident = #ident.downcast_ref::<#ty>().unwrap(); },
                (Some(_), Some(_), ident) => quote! { let #ident = #ident.downcast_mut::<#ty>().unwrap(); },
            }).collect(); //.intersperse_with(|| Comma::default().into_token_stream()).collect();

            match input.closure.body.as_ref() {
                Expr::Block(b) => {
                    let stmts = &b.block.stmts;
                    quote! { 
                        <#ty as PhysicalType>::DATATYPE => { #( #let_stmts )* #( #stmts )* }
                    }
                },
                block => {
                    quote! {
                        <#ty as PhysicalType>::DATATYPE => { #( #let_stmts )* #block }
                    }
                },
            }
        })
    }).collect();

    let dtype_expr = match input.dtype {
        Some(input_dtype) => quote! { #input_dtype },
        None => {
            let first_ident = first_ident.expect("Expected a dtype or at least one binding");
            quote! { #first_ident.dtype() }
        }
    };

    arms.push(quote! {
        _ => std::panic::panic_any(crate::error::ArrayError::type_err(format!("Unsupported operation for type '{}'", #dtype_expr)))
    });

    quote! {
        match #dtype_expr {
            #( #arms ),*,
        }
    }.into()
}

#[proc_macro_attribute]
pub fn forward_val_to_ref(_attr: proc_macro::TokenStream, item: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let item = parse_macro_input!(item as ItemImpl);

    let (trait_self, lifetime_to_remove) = match item.self_ty.as_ref() {
        Type::Reference(ref_) => (ref_.elem.as_ref().clone(), ref_.lifetime.clone()),
        ty => {
            return quote_spanned! {
                ty.span() => compile_error!("Trait must be implemented for a reference type");
            }.into();
        }
    };

    match &item.trait_ {
        Some((None, _path, _for)) => (),
        _ => {
            return quote_spanned! {
                item.span() => compile_error!("Expected a trait implementation");
            }.into();
        }
    };

    fn _map_fn(f: ImplItemFn) -> ImplItemFn {
        let name = &f.sig.ident;
        let inputs: Vec<_> = f.sig.inputs.iter().filter_map(|arg| if let FnArg::Typed(typed) = arg {
            Some(typed.pat.clone())
        } else { None }).collect();
        let block: Block = parse_quote! { { (&self).#name( #( #inputs ),* ) } };
        ImplItemFn { block, ..f }
    }

    let trait_items: Vec<ImplItem> = item.items.clone().into_iter().map(|item| match item {
        ImplItem::Fn(f) => ImplItem::Fn(_map_fn(f)),
        item => item,
    }).collect();

    let new_impl = Item::Impl(ItemImpl {
        self_ty: Box::new(trait_self),
        items: trait_items,
        generics: Generics {
            params: item.generics.params.iter().filter(|&param| ! match (param, &lifetime_to_remove) {
                (GenericParam::Lifetime(l1), Some(l2)) => l1.lifetime == *l2,
                _ => false,
            }).cloned().collect(),
            ..item.generics.clone()
        },
        ..item.clone()
    });

    quote! {
        #item
        #new_impl
    }.into()
}

#[proc_macro_attribute]
pub fn impl_for_types(_attr: proc_macro::TokenStream, item: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let item = parse_macro_input!(item as ItemImpl);

    let types: Vec<Type> = match item.self_ty.as_ref() {
        Type::Paren(ty) => vec![*ty.elem.clone()],
        Type::Tuple(tup) => tup.elems.iter().cloned().collect(),
        ty => {
            return quote_spanned! {
                ty.span() => compile_error!("Trait must be implemented for a tuple of types");
            }.into();
        }
    };

    let new_impls: Vec<ItemImpl> = types.into_iter().map(|ty| {
        let mut item = item.clone();
        item.self_ty = ty.into();
        item
    }).collect();

    quote! { #( #new_impls )* }.into()
}

